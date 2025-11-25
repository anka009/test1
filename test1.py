# -------------------- Imports --------------------
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
from pathlib import Path
from sklearn.cluster import DBSCAN

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, min_dist=6):
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def apply_dbscan(points, eps, min_samples):
    if len(points) == 0:
        return points
    pts = np.array(points)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    clustered = {}
    for lbl, p in zip(labels, pts):
        clustered.setdefault(lbl, []).append(p)
    out = []
    for plist in clustered.values():
        arr = np.array(plist)
        center = arr.mean(axis=0)
        out.append((int(center[0]), int(center[1])))
    return out

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler – OD/Deconv", layout="wide")
st.title("🧬 Zellkern-Zähler – OD/Deconv (iterative Kalibrierung)")

# -------------------- Session State --------------------
default_lists = [
    "aec_cal_points", "hema_cal_points", "bg_cal_points",
    "aec_auto", "hema_auto",
    "manual_aec", "manual_hema",
    "aec_vec", "hema_vec", "bg_vec",
    "last_file", "disp_width", "last_auto_run"
]
for key in default_lists:
    if key not in st.session_state:
        if key in ["aec_vec", "hema_vec", "bg_vec"]:
            st.session_state[key] = None
        elif key == "disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = []

# per-mode first-click-ignore flags
for flag in ["aec_first_ignore", "hema_first_ignore", "bg_first_ignore"]:
    if flag not in st.session_state:
        st.session_state[flag] = True

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "aec_auto", "hema_auto", "manual_aec", "manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_vec", "hema_vec", "bg_vec"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = st.sidebar.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5, step=1)
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 500, 50)
cluster_eps = st.sidebar.number_input("Cluster-Radius (eps)", 1, 500, 25)
cluster_min_samples = st.sidebar.number_input("Min. Punkte pro Cluster", 1, 20, 1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)
calib_patch_radius = st.sidebar.slider("🎯 Kalibrierungsradius (px)", 1, 15, 5)
iter_blend = st.sidebar.slider("🔄 Iterativer Blend für Kalibrierung", 0.0, 1.0, 0.6, step=0.05)
min_points_calib = st.sidebar.slider("🧮 Min. Punkte für Kalibrierung", 1, 10, 3)

# -------------------- Modus --------------------
mode = st.sidebar.radio(
    "Modus",
    ["Kalibrierpunkt setzen", "Manuell hinzufügen", "Punkt löschen"],
    index=0
)

# -------------------- Klicklogik --------------------
coords = streamlit_image_coordinates(Image.fromarray(image_disp), width=DISPLAY_WIDTH)

if coords is not None:
    x, y = int(coords["x"]), int(coords["y"])
    if mode == "Punkt löschen":
        for key in ["aec_cal_points","hema_cal_points","bg_cal_points","manual_aec","manual_hema"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),circle_radius)]
    elif mode == "Kalibrierpunkt setzen":
        # Beispiel: alle Punkte gehen in bg_cal_points, kann nach Wunsch getrennt werden
        st.session_state.bg_cal_points.append((x,y))
    elif mode == "Manuell hinzufügen":
        st.session_state.manual_aec.append((x,y))  # oder hema, je nach Bedarf

# Deduplication
for k in ["aec_cal_points","hema_cal_points","bg_cal_points","manual_aec","manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius//2))

# -------------------- OD/Deconv Auto-Kalibrierung --------------------
def normalize_vector(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def median_od_vector_from_points(img, points, radius=5):
    patch_vectors = []
    for x, y in points:
        x_min = max(0, x-radius)
        x_max = min(img.shape[1], x+radius)
        y_min = max(0, y-radius)
        y_max = min(img.shape[0], y+radius)
        patch = img[y_min:y_max, x_min:x_max]
        if patch.size > 0:
            od = -np.log((patch.astype(float)+1)/255)
            od_mean = od.reshape(-1,3).mean(axis=0)
            patch_vectors.append(od_mean)
    if patch_vectors:
        return normalize_vector(np.median(np.vstack(patch_vectors), axis=0))
    return None

def blend_vectors(old, new, weight_new=0.6):
    if old is None: return new
    if new is None: return old
    return normalize_vector((1-weight_new)*old + weight_new*new)

calibrated_any = False

if len(st.session_state.bg_cal_points) >= min_points_calib:
    vec_bg = median_od_vector_from_points(image_disp, st.session_state.bg_cal_points, radius=calib_patch_radius)
    st.session_state.bg_vec = blend_vectors(st.session_state.bg_vec, vec_bg, weight_new=iter_blend)
    st.session_state.bg_cal_points = []
    calibrated_any = True

if calibrated_any:
    st.session_state.last_auto_run +=1
    st.success("✅ OD/Deconv Auto-Kalibrierung durchgeführt.")

# -------------------- Gesamter Kern-Count --------------------
all_points = st.session_state.aec_auto + st.session_state.hema_auto + st.session_state.manual_aec + st.session_state.manual_hema
st.markdown(f"### 📊 Gesamtanzahl erkannter Kerne: {len(all_points)}")

# Optional: Ergebnisbild
result_img = image_disp.copy()
for (x,y) in all_points:
    cv2.circle(result_img,(x,y),circle_radius,(0,0,255),2)

st.image(result_img, caption="Erkannte Kerne (auto+manuell)", use_column_width=True)
