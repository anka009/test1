# canvas_v3_cnn_finetune.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from streamlit_image_coordinates import streamlit_image_coordinates
import os

st.set_page_config(page_title="Iterative Kern-Zählung — CNN Finetune V3", layout="wide")
st.title("🧬 Iterative Kern-Zählung — CNN Finetune V3")

# -------------------- Konstanten --------------------
MODEL_INPUT_SIZE = 64  # kleineres Input für schnelleres Finetuning
PRESET_COLORS = [(220,20,60),(0,128,0),(30,144,255),(255,165,0),(148,0,211),(0,255,255)]

# -------------------- Hilfsfunktionen --------------------
def build_unet(input_shape=(MODEL_INPUT_SIZE,MODEL_INPUT_SIZE,1), base_filters=16, depth=3):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []
    for d in range(depth):
        f = base_filters*(2**d)
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
    f = base_filters*(2**depth)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    for d in reversed(range(depth)):
        f = base_filters*(2**d)
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x,skips[d]])
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    outputs = layers.Conv2D(1,1,activation='sigmoid')(x)
    return models.Model(inputs,outputs)

def prepare_patch(patch, target_size=(MODEL_INPUT_SIZE,MODEL_INPUT_SIZE)):
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    patch_gray = patch_gray.astype(np.float32)/255.0
    return patch_gray[np.newaxis,...,np.newaxis]

def extract_patches(img, points, radius=16):
    patches = []
    for x,y in points:
        y0=max(0,y-radius)
        y1=min(img.shape[0],y+radius)
        x0=max(0,x-radius)
        x1=min(img.shape[1],x+radius)
        patch = img[y0:y1,x0:x1]
        patches.append(prepare_patch(patch))
    return np.vstack(patches) if patches else np.zeros((0,MODEL_INPUT_SIZE,MODEL_INPUT_SIZE,1))

def postprocess_mask(mask, min_size=3):
    mask_bool = mask>0
    mask_clean = remove_small_objects(mask_bool, min_size=min_size)
    mask_clean = remove_small_holes(mask_clean, area_threshold=min_size)
    return mask_clean.astype(np.uint8)

def find_centers_from_mask(mask):
    lbl = label(mask)
    centers = []
    for r in regionprops(lbl):
        cy,cx = r.centroid
        centers.append((int(round(cx)),int(round(cy))))
    return centers

def dedup_new_points(candidates, existing, min_dist=5):
    out=[]
    for c in candidates:
        if not any(np.linalg.norm(np.array(c)-np.array(e))<min_dist for e in existing):
            out.append(c)
    return out

# -------------------- Session State --------------------
for k in ["groups","all_points","last_file","disp_width","history","model","annot_points"]:
    if k not in st.session_state:
        if k in ["groups","all_points","history","annot_points"]:
            st.session_state[k]=[]
        elif k=="disp_width":
            st.session_state[k]=1000
        else:
            st.session_state[k]=None

# -------------------- Upload & Parameter --------------------
uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.history=[]
    st.session_state.annot_points=[]
    st.session_state.last_file=uploaded_file.name

col1,col2 = st.columns([2,1])
with col2:
    st.sidebar.markdown("### Parameter")
    circle_radius = st.sidebar.slider("Marker-Radius (px, Display)",1,12,5)
    detection_threshold = st.sidebar.slider("Threshold CNN-Detektion",0.01,0.99,0.5,0.01)
    stride = st.sidebar.slider("Patch-Stride (px)",16,128,32,16)
    radius_patch = st.sidebar.slider("Patch-Radius für Annotation",4,32,16)
    min_object_size = st.sidebar.slider("Min. Objektgröße (px)",1,50,3)
    model_file = st.sidebar.text_input("Pfad zu trainiertem Modell (H5)",value="trained_unet.h5")
with col1:
    DISPLAY_WIDTH = st.slider("Anzeige-Breite (px)",300,1600,st.session_state.disp_width)
    st.session_state.disp_width=DISPLAY_WIDTH

# -------------------- Prepare images --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH/float(W_orig)
H_disp=int(round(H_orig*scale))
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,H_disp),interpolation=cv2.INTER_AREA)

# -------------------- Draw existing points --------------------
display_canvas = image_disp.copy()
for i,g in enumerate(st.session_state.groups):
    col=tuple(int(x) for x in g.get("color",PRESET_COLORS[i%len(PRESET_COLORS)]))
    for (x_orig,y_orig) in g["points"]:
        x_disp=int(round(x_orig*scale))
        y_disp=int(round(y_orig*scale))
        cv2.circle(display_canvas,(x_disp,y_disp),circle_radius,col,-1)

coords = streamlit_image_coordinates(Image.fromarray(display_canvas),
                                     key=f"clickable_image_cnn_finetune_{st.session_state.last_file}",
                                     width=DISPLAY_WIDTH)

# -------------------- Load / Build Model --------------------
if st.session_state.model is None:
    if os.path.exists(model_file):
        st.session_state.model = load_model(model_file)
        st.success(f"CNN-Modell geladen: {model_file}")
    else:
        st.session_state.model = build_unet()
        st.warning("Kein trainiertes Modell gefunden. Neues U-Net erstellt.")

# -------------------- Sidebar Aktionen --------------------
mode = st.sidebar.radio("Aktion",["Annotation & Fine-Tune","Auto-Kerne erkennen","Manuelle Korrektur (Klick)","Punkt löschen","Undo letzte Aktion"])
if st.sidebar.button("Reset (Alle Gruppen)"):
    st.session_state.history.append(("reset",{"groups":st.session_state.groups.copy(),"all_points":st.session_state.all_points.copy()}))
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.annot_points=[]
    st.success("Zurückgesetzt.")

# -------------------- Annotation & Fine-Tune --------------------
if coords and mode=="Annotation & Fine-Tune":
    x_disp,y_disp=int(coords["x"]),int(coords["y"])
    x_orig=int(round(x_disp/scale))
    y_orig=int(round(y_disp/scale))
    st.session_state.annot_points.append((x_orig,y_orig))
    st.success(f"Annotierter Punkt: {len(st.session_state.annot_points)}")

if st.button("Fine-Tune CNN (kurz)"):
    if st.session_state.annot_points:
        X_train = extract_patches(image_orig, st.session_state.annot_points, radius=radius_patch)
        y_train = np.ones_like(X_train)
        st.session_state.model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
        st.session_state.model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)
        st.success("CNN feinjustiert auf annotierten Punkten.")
    else:
        st.info("Keine annotierten Punkte vorhanden.")

# -------------------- Auto-Kerne erkennen --------------------
if st.button("Auto-Kerne erkennen"):
    mask_total = np.zeros((H_orig,W_orig),dtype=np.float32)
    for y in range(0,H_orig,stride):
        for x in range(0,W_orig,stride):
            y0=y
            x0=x
            y1=min(y+MODEL_INPUT_SIZE,H_orig)
            x1=min(x+MODEL_INPUT_SIZE,W_orig)
            patch=image_orig[y0:y1,x0:x1]
            batch=prepare_patch(patch,target_size=(MODEL_INPUT_SIZE,MODEL_INPUT_SIZE))
            pred=st.session_state.model.predict(batch,verbose=0)[0,...,0]
            pred_resized=cv2.resize(pred,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
            mask_total[y0:y1,x0:x1]=np.maximum(mask_total[y0:y1,x0:x1],pred_resized)
    mask_bin=(mask_total>detection_threshold).astype(np.uint8)
    mask_clean=postprocess_mask(mask_bin,min_size=min_object_size)
    centers=find_centers_from_mask(mask_clean)
    new_centers=dedup_new_points(centers, st.session_state.all_points, min_dist=5)
    if new_centers:
        color=PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]
        st.session_state.groups.append({"points":new_centers,"color":color})
        st.session_state.all_points.extend(new_centers)
        st.success(f"Automatisch erkannte Kerne: {len(new_centers)}")
    else:
        st.info("Keine neuen Kerne erkannt.")

# -------------------- Manuelle Korrektur --------------------
if coords and mode=="Manuelle Korrektur (Klick)":
    x_disp,y_disp=int(coords["x"]),int(coords["y"])
    x_orig=int(round(x_disp/scale))
    y_orig=int(round(y_disp/scale))
    color=PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]
    st.session_state.groups.append({"points":[(x_orig,y_orig)],"color":color})
    st.session_state.all_points.append((x_orig,y_orig))
    st.success("Punkt hinzugefügt.")

if coords and mode=="Punkt löschen":
    x_disp,y_disp=int(coords["x"]),int(coords["y"])
    x_orig=int(round(x_disp/scale))
    y_orig=int(round(y_disp/scale))
    removed=[]
    new_all=[]
    for p in st.session_state.all_points:
        if np.linalg.norm(np.array(p)-np.array([x_orig,y_orig]))<5:
            removed.append(p)
        else:
            new_all.append(p)
    if removed:
        st.session_state.history.append(("delete_points",{"removed":removed}))
        st.session_state.all_points=new_all
        for g in st.session_state.groups:
            g["points"]=[p for p in g["points"] if p not in removed]
        st.success(f"{len(removed)} Punkt(e) gelöscht.")
    else:
        st.info("Kein Punkt in der Nähe gefunden.")

if st.sidebar.button("Undo letzte Aktion"):
    if st.session_state.history:
        action,payload=st.session_state.history.pop()
        if action=="reset":
            st.session_state.groups=payload["groups"]
            st.session_state.all_points=payload["all_points"]
            st.success("Reset rückgängig gemacht.")
    else:
        st.info("Keine Aktion zum Rückgängig machen.")

# -------------------- Ergebnis-Anzeige & Export --------------------
st.markdown("## Ergebnisse")
colA,colB=st.columns([2,1])
with colA:
    st.image(display_canvas,caption="Gezählte Kerne (Gruppenfarben)",use_column_width=True)
with colB:
    st.markdown("### Zusammenfassung")
    st.write(f"🔹 Gruppen gesamt: {len(st.session_state.groups)}")
    for i,g in enumerate(st.session_state.groups):
        st.write(f"• Gruppe {i+1}: {len(g['points'])} neue Kerne")
    st.markdown(f"**Gesamt (unique Kerne): {len(st.session_state.all_points)}**")
    if st.session_state.all_points:
        rows=[]
        for i,g in enumerate(st.session_state.groups):
            for (x_orig,y_orig) in g["points"]:
                x_disp=int(round(x_orig*scale))
                y_disp=int(round(y_orig*scale))
                rows.append({"Group":i+1,"X_display":x_disp,"Y_display":y_disp,"X_original":x_orig,"Y_original":y_orig})
        df=pd.DataFrame(rows)
        st.download_button("📥 CSV exportieren",df.to_csv(index=False).encode("utf-8"),
                           file_name="kern_gruppen_v3_finetune.csv",mime="text/csv")
