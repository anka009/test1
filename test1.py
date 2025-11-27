# canvas_iterative_deconv_v3_cnn.py
"""
V3-CNN: Full Streamlit app for nucleus detection using a Small U-Net
Features (based on your choices):
- Small U-Net (M-sized ~1-3M params)
- Train / Fine-tune inside the app from user-uploaded images+masks
- Live inference on uploaded images with intelligent tiling (auto-scaling)
- Auto-Threshold (Otsu on predicted probability) + Auto-Postprocessing (small-object removal)
- Auto-Calibration (simple intensity/stain normalization per-image)
- Manual correction UI: click to add/remove centers; undo history
- CSV export of detected centers

Notes:
- Requires TensorFlow (>=2.x) for training/inference. The app will show a fallback message if TF is not installed.
- Uses predictable memory-friendly tiling, batching and caching of predictions.
- Model and weights are saved locally (model.h5) after training and loaded for inference.

"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import os
import math
import io
import tempfile
import json
from typing import List, Tuple

# Optional heavy deps
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
except Exception:
    TF_AVAILABLE = False

# Utilities
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

# ----- Model definition (Small U-Net, M-size) -----
def build_unet(input_shape=(256,256,1), base_filters=32, depth=4):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []
    # Encoder
    for d in range(depth):
        f = base_filters * (2**d)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
    # Bottleneck
    f = base_filters * (2**depth)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    # Decoder
    for d in reversed(range(depth)):
        f = base_filters * (2**d)
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skips[d]])
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

# ----- Image utilities: tiling, normalization, calibration -----

def reinhard_normalize(src, target_mean=0.5, eps=1e-8):
    """Simple intensity normalization to a target mean (per-image)."""
    arr = src.astype(np.float32) / 255.0
    mean = arr.mean()
    scale = (target_mean + eps) / (mean + eps)
    dst = np.clip(arr * scale, 0.0, 1.0)
    return (dst * 255.0).astype(np.uint8)


def image_to_patches(img: np.ndarray, patch_size:int, overlap:int) -> Tuple[List[np.ndarray], List[Tuple[int,int]]]:
    H, W = img.shape[:2]
    stride = patch_size - overlap
    patches = []
    coords = []
    for y in range(0, H, stride):
        if y + patch_size > H:
            y = max(0, H - patch_size)
        for x in range(0, W, stride):
            if x + patch_size > W:
                x = max(0, W - patch_size)
            patch = img[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                # pad
                pad = np.zeros((patch_size, patch_size, img.shape[2]), dtype=img.dtype)
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad
            patches.append(patch)
            coords.append((x,y))
        # break if last block reached
        if y + patch_size >= H:
            break
    return patches, coords


def patches_to_image(pred_patches: List[np.ndarray], coords: List[Tuple[int,int]], out_shape: Tuple[int,int], patch_size:int, overlap:int):
    H, W = out_shape
    heat = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    for p, (x,y) in zip(pred_patches, coords):
        h = min(patch_size, H - y)
        w = min(patch_size, W - x)
        heat[y:y+h, x:x+w] += p[:h,:w]
        weight[y:y+h, x:x+w] += 1.0
    weight[weight==0] = 1.0
    heat /= weight
    return heat

# ----- Postprocessing: thresholding, small object removal, center extraction -----

def auto_threshold(prob_map: np.ndarray) -> float:
    # Otsu threshold on probability map scaled to 0..255
    u8 = np.clip((prob_map * 255.0).astype(np.uint8), 0, 255)
    try:
        _, thr = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr / 255.0
    except Exception:
        return 0.5


def postprocess_mask(prob_map: np.ndarray, thr: float=None, min_area:int=5) -> np.ndarray:
    if thr is None:
        thr = auto_threshold(prob_map)
    mask = (prob_map >= thr).astype(np.uint8)
    # remove small objects
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out_mask[labels==i] = 1
    return out_mask


def extract_centers_from_mask(mask: np.ndarray, min_area=5) -> List[Tuple[int,int]]:
    contours, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= max(1, min_area):
            M = cv2.moments(c)
            if M.get('m00', 0) != 0:
                cx = int(round(M['m10']/M['m00']))
                cy = int(round(M['m01']/M['m00']))
                centers.append((cx, cy))
    return centers

# ----- Streamlit App -----
st.set_page_config(page_title="CNN Nuclei Detector — v3 (U-Net)", layout='wide')
st.title("🧠 CNN-based Nuclei Detection — U-Net (v3)")

# Sidebar: model & data
with st.sidebar:
    st.header("Model & Data")
    model_variant = st.selectbox("U-Net base filters", [16, 32, 48], index=1, help="Base number of filters. Larger = better quality but slower.")
    input_patch = st.selectbox("Model input patch (px)", [128, 192, 256, 320], index=2)
    overlap = st.selectbox("Tile overlap (px)", [16, 32, 48, 64], index=1)
    min_area = st.number_input("Min component area (px)", min_value=1, max_value=500, value=8)
    train_epochs = st.number_input("Train epochs", min_value=1, max_value=200, value=25)
    batch_size = st.number_input("Train batch size", min_value=1, max_value=32, value=8)

st.sidebar.markdown("---")
with st.sidebar.expander("Advanced intelligent features"):
    auto_scaling = st.checkbox("Auto-Scaling (tile size adapt to image)", value=True)
    auto_post = st.checkbox("Auto-Postprocessing (small-object removal)", value=True)
    auto_calib = st.checkbox("Auto-Calibration (simple intensity norm)", value=True)
    auto_thresh = st.checkbox("Auto-Threshold (Otsu)", value=True)

# Model load / build
MODEL_PATH = "model_unet_v3.h5"
model = None
if TF_AVAILABLE:
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            st.sidebar.success(f"Model geladen: {MODEL_PATH}")
        except Exception as e:
            st.sidebar.warning(f"Fehler beim Laden des Modells: {e}. Neu bauen.")
            model = build_unet(input_shape=(input_patch, input_patch, 1), base_filters=model_variant, depth=4)
    else:
        model = build_unet(input_shape=(input_patch, input_patch, 1), base_filters=model_variant, depth=4)
else:
    st.sidebar.error("TensorFlow nicht installiert. Installiere tensorflow, um Training/Inferenz zu nutzen.")

# Data upload for training
st.markdown("## 1) Dataset (Train/Fine-tune)")
st.info("Upload pairs of images + masks. Masks should be binary images with same filename + suffix '_mask' or uploaded in the same order.")
col_u1, col_u2 = st.columns(2)
with col_u1:
    imgs = st.file_uploader("Upload input images (multiple)", type=["png","jpg","tif","tiff"], accept_multiple_files=True)
with col_u2:
    masks = st.file_uploader("Upload masks (binary) (multiple)", type=["png","jpg","tif","tiff"], accept_multiple_files=True)

train_btn = st.button("Start Training / Fine-tune")

# Inference area
st.markdown("## 2) Inference & Manual Correction")
uploaded = st.file_uploader("Upload image for inference (single)", type=["png","jpg","tif","tiff"]) 

# Session state for UI interactions
for k in ["train_data", "pred_cache", "manual_points", "history", "last_infer_file"]:
    if k not in st.session_state:
        if k in ["train_data", "manual_points", "history"]:
            st.session_state[k] = []
        else:
            st.session_state[k] = None

# ----- Training pipeline (simple) -----
if train_btn:
    if not TF_AVAILABLE:
        st.error("TensorFlow nicht verfügbar — Training nicht möglich.")
    else:
        # Basic validation: require equal number of images and masks
        if not imgs or not masks or len(imgs) != len(masks):
            st.error("Bitte gleiche Anzahl von Bildern und Masken hochladen (oder paired filenames).")
        else:
            # load all into memory (careful with many large files)
            X = []
            Y = []
            with st.spinner("Lade Trainingsdaten..."):
                for f_img, f_mask in zip(imgs, masks):
                    im = Image.open(f_img).convert('RGB')
                    mk = Image.open(f_mask).convert('L')
                    im_a = np.array(im)
                    mk_a = np.array(mk)
                    # auto-calibration
                    if auto_calib:
                        im_a = reinhard_normalize(im_a)
                    # convert mask to binary
                    mk_bin = (mk_a > 127).astype(np.uint8)
                    # tile or resize to input_patch
                    im_res = cv2.resize(cv2.cvtColor(im_a, cv2.COLOR_RGB2GRAY), (input_patch, input_patch), interpolation=cv2.INTER_AREA)
                    mk_res = cv2.resize(mk_bin, (input_patch, input_patch), interpolation=cv2.INTER_NEAREST)
                    X.append(im_res[..., None].astype(np.float32) / 255.0)
                    Y.append(mk_res[..., None].astype(np.float32))
            X = np.stack(X, axis=0)
            Y = np.stack(Y, axis=0)
            st.write(f"Trainingssamples: {X.shape[0]} | Patch size: {input_patch}")

            # compile model with dice+binary crossentropy
            def dice_loss(y_true, y_pred, smooth=1e-6):
                y_true_f = tf.reshape(y_true, [-1])
                y_pred_f = tf.reshape(y_pred, [-1])
                intersection = tf.reduce_sum(y_true_f * y_pred_f)
                return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

            def combined_loss(y_true, y_pred):
                bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                d = dice_loss(y_true, y_pred)
                return bce + d

            model.compile(optimizer=optimizers.Adam(1e-4), loss=combined_loss, metrics=["accuracy"]) 
            with st.spinner("Training läuft — das kann einige Minuten dauern..."):
                history = model.fit(X, Y, epochs=int(train_epochs), batch_size=int(batch_size), validation_split=0.1)
            # save model
            model.save(MODEL_PATH)
            st.success("Training abgeschlossen — Modell gespeichert.")

# ----- Inference pipeline -----
if uploaded:
    img = np.array(Image.open(uploaded).convert('RGB'))
    H, W = img.shape[:2]
    st.write(f"Bild geladen: {uploaded.name} — Größe: {W}x{H}")

    # auto-calibration
    if auto_calib:
        img_proc = reinhard_normalize(img)
    else:
        img_proc = img.copy()

    # auto-scaling: choose patch_size based on image if enabled
    patch_size = input_patch
    if auto_scaling:
        # heuristic: reduce patch_size for small images, increase for big
        max_dim = max(H, W)
        if max_dim > 3000:
            patch_size = min(512, input_patch + 128)
        elif max_dim < 800:
            patch_size = max(128, input_patch - 64)

    # prepare patches
    patches, coords = image_to_patches(cv2.cvtColor(img_proc, cv2.COLOR_RGB2GRAY)[...,None], patch_size, overlap)
    # normalize patches
    Xp = np.stack([p.astype(np.float32)/255.0 for p in patches], axis=0)

    # batch predict
    preds = []
    if TF_AVAILABLE and model is not None:
        B = 8
        for i in range(0, len(Xp), B):
            batch = Xp[i:i+B]
            pred = model.predict(batch)
            preds.extend([p[...,0] for p in pred])
    else:
        st.error("TensorFlow/Model nicht verfügbar — Inferenz nicht möglich.")
        preds = [np.zeros((patch_size, patch_size), dtype=np.float32) for _ in patches]

    # reconstruct full probability map
    prob_map = patches_to_image(preds, coords, (H, W), patch_size, overlap)

    # threshold and postprocess
    thr = None
    if auto_thresh:
        thr = auto_threshold(prob_map)
    else:
        thr = 0.5
    mask = postprocess_mask(prob_map, thr=thr, min_area=int(min_area if auto_post else 1))
    centers = extract_centers_from_mask(mask, min_area=int(min_area if auto_post else 1))

    # cache prediction for manual correction
    st.session_state.pred_cache = {
        'file': uploaded.name,
        'prob_map': prob_map.tolist(),
        'mask': mask.tolist(),
        'centers': centers
    }
    st.session_state.last_infer_file = uploaded.name

    # display overlay & interactive correction
    display_width = st.slider("Anzeige-Breite (px)", 300, 1600, 1000)
    scale = display_width / float(W)
    H_disp = int(round(H * scale))
    img_disp = cv2.resize(img, (display_width, H_disp), interpolation=cv2.INTER_AREA)
    overlay = img_disp.copy()
    # draw centers
    for (x,y) in centers:
        x_d = int(round(x * scale)); y_d = int(round(y * scale))
        cv2.circle(overlay, (x_d, y_d), 6, (0,255,0), -1)
    # clickable image
    from streamlit_image_coordinates import streamlit_image_coordinates
    coords_click = streamlit_image_coordinates(Image.fromarray(overlay), key=f"cnn_infer_{uploaded.name}", width=display_width)

    st.markdown("### Ergebnisse")
    colA, colB = st.columns([2,1])
    with colA:
        st.image(overlay, caption="Prediction overlay (click to add/delete)", use_column_width=True)
    with colB:
        st.write(f"Detected centers: {len(centers)}")
        st.write(f"Auto-threshold: {thr:.3f}")
        if st.button("Export CSV (centers)"):
            rows = []
            for (x,y) in centers:
                rows.append({"X_original": int(x), "Y_original": int(y)})
            df = pd.DataFrame(rows)
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), file_name='cnn_centers.csv', mime='text/csv')

    # manual correction handling
    if coords_click:
        x_d = int(coords_click['x']); y_d = int(coords_click['y'])
        x_o = int(round(x_d / scale)); y_o = int(round(y_d / scale))
        # toggle: if near existing -> delete, else add
        found = None
        for c in centers:
            if math.hypot(c[0]-x_o, c[1]-y_o) <= 8:
                found = c
                break
        if found:
            centers.remove(found)
            st.session_state.history.append(('delete_point', {'point': found}))
            st.success('Punkt gelöscht')
        else:
            centers.append((x_o, y_o))
            st.session_state.history.append(('add_point', {'point': (x_o,y_o)}))
            st.success('Punkt hinzugefügt')
        # update cache and redisplay
        st.session_state.pred_cache['centers'] = centers
        # redraw overlay
        overlay2 = img_disp.copy()
        for (x,y) in centers:
            cv2.circle(overlay2, (int(round(x*scale)), int(round(y*scale))), 6, (0,255,0), -1)
        st.image(overlay2, caption='Updated overlay', use_column_width=True)

# Undo button
if st.button('Undo letzte Aktion'):
    if st.session_state.history:
        action, payload = st.session_state.history.pop()
        if action == 'delete_point':
            p = tuple(payload['point'])
            st.session_state.pred_cache['centers'].append(p)
            st.success('Undo: Punkt wiederhergestellt')
        elif action == 'add_point':
            p = tuple(payload['point'])
            st.session_state.pred_cache['centers'] = [c for c in st.session_state.pred_cache['centers'] if c != p]
            st.success('Undo: hinzugefügter Punkt entfernt')
    else:
        st.info('Keine Aktion zum Rückgängig machen')

st.markdown('---')
st.caption('Hinweis: Diese Version führt die komplette Kern-Erkennung via CNN durch (U-Net). Wenn du möchtest, kann ich jetzt noch: 1) performance-optimieren (GPU-Batching, mixed precision), 2) Tile-Visualizer, 3) Auto-finetune-on-corrections.')
