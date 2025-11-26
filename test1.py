# canvas_iterative_deconv_v2_hybrid.py
# Hybrid-Version: Konturen + LoG-Blob-Detection integriert
# (vollständig ersetzt & integriert)

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.feature import blob_log
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv + Hybrid)", layout="wide")
st.title("🧬 Iterative Kern-Zählung — Hybrid-Version (Contour + LoG)")

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=6):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_new_points(candidates, existing, min_dist=6):
    out = []
    for c in candidates:
        if not any(is_near(c, e, min_dist) for e in existing):
            out.append(c)
    return out

def extract_patch(img, x, y, radius=5):
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    return img[y_min:y_max, x_min:x_max]

def median_od_vector_from_patch(patch, eps=1e-6):
    if patch is None or patch.size == 0:
        return None
    patch = patch.astype(np.float32)
    OD = -np.log(np.clip((patch + eps) / 255.0, 1e-8, 1.0))
    vec = np.median(OD.reshape(-1, 3), axis=0)
    n = np.linalg.norm(vec)
    if n <= 1e-8 or np.any(np.isnan(vec)):
        return None
    return (vec / n).astype(np.float32)

def normalize_vector(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return (v / n) if n > 1e-12 else v

def make_stain_matrix(target_vec, hema_vec, bg_vec=None):
    t = normalize_vector(target_vec)
    h = normalize_vector(hema_vec)
    if bg_vec is None:
        bg = np.cross(t, h)
        if np.linalg.norm(bg) < 1e-6:
            if abs(t[0]) > 0.1 or abs(t[1]) > 0.1:
                bg = np.array([t[1], -t[0], 0])
            else:
                bg = np.array([0, t[2], -t[1]])
        bg = normalize_vector(bg)
    else:
        bg = normalize_vector(bg_vec)
    M = np.column_stack([t, h, bg]).astype(np.float32)
    M += np.eye(3) * 1e-8
    return M

def deconvolve(img_rgb, M):
    img = img_rgb.astype(np.float32)
    OD = -np.log(np.clip((img + 1e-6) / 255.0, 1e-8, 1.0)).reshape(-1, 3)
    try:
        pinv = np.linalg.pinv(M)
        C = (pinv @ OD.T).T
    except Exception:
        return None
    return C.reshape(img_rgb.shape)

def detect_contours(channel, threshold=0.2, min_area=8):
    arr = np.maximum(channel.astype(np.float32), 0)
    vmin, vmax = np.percentile(arr, [2, 99.5])
    if vmax - vmin < 1e-5:
        return []
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(u8, (5, 5), 0)
    block = 35 if min(arr.shape) > 100 else 15
    if block % 2 == 0:
        block += 1
    try:
        mask = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, block, -2)
    except:
        _, mask = cv2.threshold(blur, int(threshold * 255), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers

# NEW: Blob-LoG detection
def detect_blob_log(channel, min_sigma=1, max_sigma=6, num_sigma=8, threshold=0.03):
    arr = np.maximum(channel.astype(float), 0)
    vmin, vmax = np.percentile(arr, [2, 99.5])
    if vmax - vmin < 1e-5:
        return []
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    blobs = blob_log(norm, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold)
    centers = [(int(round(y)), int(round(x))) for y, x, s in blobs]
    return centers

# -------------------- Session State --------------------
for k in ["groups", "all_points", "last_file", "disp_width", "C_cache", "last_M_hash", "history"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k in ["groups", "all_points", "history"] else None
if st.session_state.disp_width is None:
    st.session_state.disp_width = 1000

# -------------------- UI --------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "tif", "tiff"])
if not uploaded_file:
    st.stop()

# initialize for new file
if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups = []
    st.session_state.all_points = []
    st.session_state.C_cache = None
    st.session_state.history = []
    st.session_state.last_file = uploaded_file.name

col1, col2 = st.columns([2, 1])
with col2:
    st.sidebar.markdown("### Parameter")
    calib_radius = st.sidebar.slider("Kalibrier-Radius", 1, 30, 5)
    threshold_ui = st.sidebar.slider("Threshold", 0.01, 0.9, 0.2)
    min_area_display = st.sidebar.number_input("Min. Fläche (px, Display)", 1, 1000, 8)
    dedup_display = st.sidebar.slider("Dedup (px Display)", 1, 40, 6)
    circle_r = st.sidebar.slider("Marker-Radius", 1, 12, 5)

    st.sidebar.markdown("### Blob-Parameter")
    blob_thr = st.sidebar.slider("Blob Threshold", 0.001, 0.2, 0.03)
    blob_sigma = st.sidebar.slider("Blob max_sigma", 2, 15, 6)

    st.sidebar.markdown("### Startvektoren")
    hv = st.sidebar.text_input("Hema", "0.65,0.70,0.29")
    cv = st.sidebar.text_input("Chromogen", "0.27,0.57,0.78")
    try:
        hema_vec = np.array([float(x) for x in hv.split(',')])
        chrom_vec = np.array([float(x) for x in cv.split(',')])
    except:
        hema_vec = np.array([0.65, 0.70, 0.29])
        chrom_vec = np.array([0.27, 0.57, 0.78])

with col1:
    DISPLAY_WIDTH = st.slider("Anzeige-Breite", 300, 1600, st.session_state.disp_width)
    st.session_state.disp_width = DISPLAY_WIDTH

# -------------------- Image scaling --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H0, W0 = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W0
H_disp = int(round(H0 * scale))
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, H_disp))
area_scale = 1 / (scale * scale)
min_area_orig = max(1, int(min_area_display * area_scale))
dedup_orig = dedup_display / scale

# -------------------- Draw Points --------------------
PRESET_COLORS = [(220,20,60),(0,128,0),(30,144,255),(255,165,0),(148,0,211),(0,255,255)]
display_canvas = image_disp.copy()
for i, g in enumerate(st.session_state.groups):
    col = PRESET_COLORS[i % len(PRESET_COLORS)]
    for (xo, yo) in g["points"]:
        xd, yd = int(xo*scale), int(yo*scale)
        cv2.circle(display_canvas, (xd,yd), circle_r, col, -1)

coords = streamlit_image_coordinates(Image.fromarray(display_canvas),
                                    key=f"img_{uploaded_file.name}", width=DISPLAY_WIDTH)

# -------------------- Mode --------------------
mode = st.sidebar.radio("Aktion", ["Kalibriere & Zähle","Punkt löschen","Undo"])
if st.sidebar.button("Reset"):
    st.session_state.history.append(("reset",{
        "groups": st.session_state.groups.copy(),
        "all_points": st.session_state.all_points.copy()}))
    st.session_state.groups=[]
