# canvas_iterative_cnn_v3.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tensorflow.keras import layers, models
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Iterative Kern-Zählung — CNN V3", layout="wide")
st.title("🧬 Iterative Kern-Zählung — CNN V3")

# -------------------- Hilfsfunktionen --------------------
def prepare_patch_for_model(patch, target_size=(256,256)):
    """
    Wandelt beliebiges Patch in korrektes U-Net Input-Batch um:
    - Resize
    - Graustufen
    - Normalisierung
    - Batch/Channel-Dimension hinzufügen
    """
    patch_resized = cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)
    if patch_resized.ndim == 3 and patch_resized.shape[2] == 3:
        patch_gray = cv2.cvtColor(patch_resized, cv2.COLOR_RGB2GRAY)
    else:
        patch_gray = patch_resized
    patch_gray = patch_gray.astype(np.float32)/255.0
    batch = patch_gray[np.newaxis, ..., np.newaxis]
    return batch

def dedup_new_points(candidates, existing, min_dist=6):
    """Return candidates not within min_dist of existing points."""
    out = []
    for c in candidates:
        if not any(np.linalg.norm(np.array(c)-np.array(e))<min_dist for e in existing):
            out.append(c)
    return out

def build_unet(input_shape=(256,256,1), base_filters=32, depth=4):
    """Small U-Net für Zentrumserkennung"""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []
    # Encoder
    for d in range(depth):
        f = base_filters*(2**d)
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
    # Bottleneck
    f = base_filters*(2**depth)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    # Decoder
    for d in reversed(range(depth)):
        f = base_filters*(2**d)
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skips[d]])
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
        x = layers.Conv2D(f,3,padding='same',activation='relu')(x)
    outputs = layers.Conv2D(1,1,activation='sigmoid')(x)
    model = models.Model(inputs,outputs)
    return model

# -------------------- Session state --------------------
for k in ["groups","all_points","last_file","disp_width","history","model"]:
    if k not in st.session_state:
        if k in ["groups","all_points","history"]:
            st.session_state[k] = []
        elif k=="disp_width":
            st.session_state[k]=1000
        else:
            st.session_state[k]=None

# -------------------- UI: Upload + Parameter --------------------
uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.history=[]
    st.session_state.last_file=uploaded_file.name

col1,col2 = st.columns([2,1])
with col2:
    st.sidebar.markdown("### Parameter")
    circle_radius = st.sidebar.slider("Marker-Radius (px, Display)",1,12,5)
    detection_threshold = st.sidebar.slider("Threshold für CNN-Detektion",0.01,0.99,0.5,0.01)
with col1:
    DISPLAY_WIDTH = st.slider("Anzeige-Breite (px)",300,1600,st.session_state.disp_width)
    st.session_state.disp_width=DISPLAY_WIDTH

# -------------------- Prepare images --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH/float(W_orig)
H_disp = int(round(H_orig*scale))
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,H_disp),interpolation=cv2.INTER_AREA)

# -------------------- Draw existing points --------------------
display_canvas = image_disp.copy()
PRESET_COLORS=[(220,20,60),(0,128,0),(30,144,255),(255,165,0),(148,0,211),(0,255,255)]
for i,g in enumerate(st.session_state.groups):
    col=tuple(int(x) for x in g.get("color",PRESET_COLORS[i%len(PRESET_COLORS)]))
    for (x_orig,y_orig) in g["points"]:
        x_disp=int(round(x_orig*scale))
        y_disp=int(round(y_orig*scale))
        cv2.circle(display_canvas,(x_disp,y_disp),circle_radius,col,-1)

coords = streamlit_image_coordinates(Image.fromarray(display_canvas),
                                     key=f"clickable_image_cnn_{st.session_state.last_file}",
                                     width=DISPLAY_WIDTH)

# -------------------- Load or build model --------------------
if st.session_state.model is None:
    st.session_state.model=build_unet()
    st.success("CNN-Modell bereit.")

# -------------------- Sidebar actions --------------------
mode = st.sidebar.radio("Aktion",["Kerne erkennen (Klick)","Punkt löschen","Undo letzte Aktion"])
if st.sidebar.button("Reset (Alle Gruppen)"):
    st.session_state.history.append(("reset",{"groups":st.session_state.groups.copy(),"all_points":st.session_state.all_points.copy()}))
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.success("Zurückgesetzt.")

# -------------------- Click handling --------------------
if coords:
    x_disp,y_disp=int(coords["x"]),int(coords["y"])
    x_orig=int(round(x_disp/scale))
    y_orig=int(round(y_disp/scale))

    if mode=="Punkt löschen":
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
    elif mode=="Undo letzte Aktion":
        if st.session_state.history:
            action,payload=st.session_state.history.pop()
            if action=="reset":
                st.session_state.groups=payload["groups"]
                st.session_state.all_points=payload["all_points"]
                st.success("Reset rückgängig gemacht.")
        else:
            st.info("Keine Aktion zum Rückgängig machen.")
    else:
        # -------------------- CNN-Erkennung --------------------
        patch_radius=64
        x_min=max(0,x_orig-patch_radius)
        x_max=min(W_orig,x_orig+patch_radius)
        y_min=max(0,y_orig-patch_radius)
        y_max=min(H_orig,y_orig+patch_radius)
        patch=image_orig[y_min:y_max,x_min:x_max]
        batch=prepare_patch_for_model(patch)
        pred=st.session_state.model.predict(batch)[0,...,0]
        # Threshold
        pred_bin=(pred>detection_threshold).astype(np.uint8)
        # find centers
        ys,xs=np.where(pred_bin>0)
        centers_orig=[(x+x_min,y+y_min) for x,y in zip(xs,ys)]
        # dedup
        new_centers=dedup_new_points(centers_orig,st.session_state.all_points,min_dist=5)
        if new_centers:
            color=PRESET_COLORS[len(st.session_state.groups)%len(PRESET_COLORS)]
            st.session_state.groups.append({"points":new_centers,"color":color})
            st.session_state.all_points.extend(new_centers)
            st.success(f"Neue Kerne: {len(new_centers)}")
        else:
            st.info("Keine neuen Kerne erkannt.")

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
                           file_name="kern_gruppen_v3.csv",mime="text/csv")
