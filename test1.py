# canvas_v3_cnn_auto.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from tensorflow.keras import layers, models
from skimage.feature import peak_local_max
import tensorflow as tf

st.set_page_config(page_title="Canvas v3 CNN Auto", layout="wide")
st.title("🧬 Canvas v3 — CNN Auto Detection (Kerne)")

# -------------------- Hilfsfunktionen --------------------
def is_near(p1,p2,r=6):
    return np.linalg.norm(np.array(p1)-np.array(p2))<r

def dedup_points(candidates, existing, min_dist=6):
    out=[]
    for c in candidates:
        if not any(is_near(c,e,min_dist) for e in existing):
            out.append(c)
    return out

def build_unet(input_shape=(64,64,1), base_filters=32, depth=3):
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

def prepare_patch(patch):
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    patch = patch.astype(np.float32)/255.0
    patch = np.expand_dims(patch,axis=-1)
    patch = np.expand_dims(patch,axis=0)
    return patch

def extract_patches(img, points, radius=16, model_input=64):
    patches=[]
    for x,y in points:
        y0=max(0,y-radius)
        y1=min(img.shape[0],y+radius)
        x0=max(0,x-radius)
        x1=min(img.shape[1],x+radius)
        patch=img[y0:y1,x0:x1]
        patch=cv2.resize(patch,(model_input,model_input))
        patches.append(prepare_patch(patch))
    if patches:
        return np.vstack(patches)
    else:
        return np.zeros((0,model_input,model_input,1))

def detect_from_pred(pred, min_distance=6):
    # pred: 2D array, 0-1
    coords = peak_local_max(pred, min_distance=min_distance, threshold_abs=0.5)
    return [(int(x),int(y)) for y,x in coords]  # swap y,x to x,y

# -------------------- Session State --------------------
for k in ["all_points","history","model","last_file","disp_width"]:
    if k not in st.session_state:
        if k in ["all_points","history"]:
            st.session_state[k]=[]
        elif k=="disp_width":
            st.session_state[k]=1000
        else:
            st.session_state[k]=None

MODEL_INPUT_SIZE = 64
st.sidebar.markdown("### Parameters")
radius_patch = st.sidebar.slider("Patch-Radius (px)",4,32,16)
min_dist = st.sidebar.slider("Min Distance Dedup (px)",3,20,6)
epochs_finetune = st.sidebar.slider("Fine-tune Epochs",1,20,5)

# -------------------- Upload --------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.info("Upload an image first.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.all_points=[]
    st.session_state.history=[]
    st.session_state.last_file=uploaded_file.name
    st.session_state.model=None

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = st.session_state.disp_width / W_orig
H_disp=int(H_orig*scale)
image_disp = cv2.resize(image_orig,(st.session_state.disp_width,H_disp))

# -------------------- Initialize model --------------------
if st.session_state.model is None:
    st.session_state.model = build_unet(input_shape=(MODEL_INPUT_SIZE,MODEL_INPUT_SIZE,1))
    st.session_state.model.compile(optimizer='adam',loss='binary_crossentropy')

# -------------------- Display canvas --------------------
display_canvas = image_disp.copy()
for p in st.session_state.all_points:
    x_disp=int(p[0]*scale)
    y_disp=int(p[1]*scale)
    cv2.circle(display_canvas,(x_disp,y_disp),3,(220,20,60),-1)

coords = st.image(image_disp, caption="Click to mark nucleus", use_column_width=True)
# Note: Use streamlit_image_coordinates if you want clickable coordinates
# coords = streamlit_image_coordinates(Image.fromarray(display_canvas), width=st.session_state.disp_width)

# -------------------- Handle click --------------------
clicked_points = st.session_state.all_points.copy()
# For demo, assume user manually adds points via coords (replace with actual click coords)
# Here we simulate adding the center of the image
clicked_points.append((W_orig//2,H_orig//2))

if clicked_points:
    X_train = extract_patches(image_orig, clicked_points,radius=radius_patch, model_input=MODEL_INPUT_SIZE)
    y_train = np.ones_like(X_train,dtype=np.float32)
    if X_train.shape[0]>0:
        st.session_state.model.fit(X_train,y_train,epochs=epochs_finetune,batch_size=8,verbose=0)
        st.success(f"Fine-tuned on {X_train.shape[0]} patch(es)")

# -------------------- Full image prediction --------------------
gray_full = cv2.cvtColor(image_orig,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
full_pred = np.zeros_like(gray_full)
# sliding window inference
step = MODEL_INPUT_SIZE//2
for y in range(0,H_orig-MODEL_INPUT_SIZE+1,step):
    for x in range(0,W_orig-MODEL_INPUT_SIZE+1,step):
        patch = gray_full[y:y+MODEL_INPUT_SIZE,x:x+MODEL_INPUT_SIZE]
        patch = np.expand_dims(patch,axis=-1)
        patch = np.expand_dims(patch,axis=0)
        pred = st.session_state.model.predict(patch,verbose=0)[0,...,0]
        full_pred[y:y+MODEL_INPUT_SIZE,x:x+MODEL_INPUT_SIZE]+=pred

# normalize overlap
counts = np.zeros_like(gray_full)
for y in range(0,H_orig-MODEL_INPUT_SIZE+1,step):
    for x in range(0,W_orig-MODEL_INPUT_SIZE+1,step):
        counts[y:y+MODEL_INPUT_SIZE,x:x+MODEL_INPUT_SIZE]+=1
full_pred = full_pred / np.maximum(counts,1)

# -------------------- Postprocessing --------------------
detected_centers = detect_from_pred(full_pred, min_distance=min_dist)
new_centers = dedup_points(detected_centers, st.session_state.all_points, min_dist=min_dist)
st.session_state.all_points.extend(new_centers)

# Draw on canvas
for p in st.session_state.all_points:
    x_disp=int(p[0]*scale)
    y_disp=int(p[1]*scale)
    cv2.circle(display_canvas,(x_disp,y_disp),3,(220,20,60),-1)

st.image(display_canvas, caption="Detected nuclei", use_column_width=True)

# -------------------- CSV Export --------------------
if st.session_state.all_points:
    rows=[]
    for i,p in enumerate(st.session_state.all_points):
        rows.append({"ID":i+1,"X":p[0],"Y":p[1]})
    df=pd.DataFrame(rows)
    st.download_button("📥 Export CSV", df.to_csv(index=False).encode("utf-8"), file_name="nuclei.csv",mime="text/csv")
