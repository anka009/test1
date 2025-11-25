# canvas_all_nuclei.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
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

def ensure_odd(k):
    return k if k % 2 == 1 else k + 1

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

def get_centers(mask, min_area=50):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0):
                cx = int(round(M["m10"] / M["m00"]))
                cy = int(round(M["m01"] / M["m00"]))
                centers.append((cx, cy))
    return centers

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def blend_vectors(old, new, weight_new=0.6):
    if old is None:
        return new
    if new is None:
        return old
    return normalize_vector((1.0 - weight_new) * old + weight_new * new)

# Dummy-Funktion für OD-Deconvolution + Maskenerstellung
def detect_all_nuclei(img, stain_vec, min_area=50):
    """
    img: RGB image
    stain_vec: OD-Vektor (3,)
    return: binäre Maske aller Kerne
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)  # einfache Schwelle
    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (gesamt)", layout="wide")
st.title("🧬 Zellkern-Zähler – Gesamtzählung")

# -------------------- Session State --------------------
for key, default in [
    ("cal_points", []),
    ("manual_points", []),
    ("all_auto", []),
    ("stain_vec", None),
    ("last_file", None),
    ("disp_width", 1400),
    ("last_auto_run", 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.cal_points = []
    st.session_state.manual_points = []
    st.session_state.all_auto = []
    st.session_state.stain_vec = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
st.session_state.disp_width = DISPLAY_WIDTH
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur", 1, 21, 5))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 500, 50)
cluster_eps = st.sidebar.number_input("Cluster-Radius (eps)", 1, 500, 25)
cluster_min_samples = st.sidebar.number_input("Min. Punkte pro Cluster", 1, 20, 1)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)
calib_patch_radius = st.sidebar.slider("🎯 Kalibrierungsradius", 1, 15, 5)
min_points_calib = st.sidebar.slider("🧮 Minimale Punkte für Kalibrierung", 1, 10, 3)
iter_blend = st.sidebar.slider("🔄 Blendgewicht (neu)", 0.0, 1.0, 0.6, step=0.05)

# -------------------- Bildanzeige mit Klick --------------------
marked_disp = image_disp.copy()
for x, y in st.session_state.cal_points:
    cv2.circle(marked_disp, (x, y), circle_radius, (200, 0, 200), -1)
for x, y in st.session_state.manual_points:
    cv2.circle(marked_disp, (x, y), circle_radius, (0, 165, 255), -1)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp),
                                     key=f"clickable_image_{st.session_state.last_file}",
                                     width=DISPLAY_WIDTH)

# -------------------- Klicklogik --------------------
if coords is not None:
    x, y = int(coords["x"]), int(coords["y"])
    if st.sidebar.radio("Modus", ["Kalibrierpunkt setzen", "Manuell hinzufügen", "Punkt löschen"]) == "Punkt löschen":
        for key in ["cal_points","manual_points"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),circle_radius)]
    elif st.sidebar.radio("Modus", ["Kalibrierpunkt setzen", "Manuell hinzufügen", "Punkt löschen"]) == "Kalibrierpunkt setzen":
        st.session_state.cal_points.append((x,y))
    else:
        st.session_state.manual_points.append((x,y))

# Dedup
st.session_state.cal_points = dedup_points(st.session_state.cal_points, min_dist=max(4, circle_radius//2))
st.session_state.manual_points = dedup_points(st.session_state.manual_points, min_dist=max(4, circle_radius//2))

# -------------------- Auto-Kalibrierung --------------------
if len(st.session_state.cal_points) >= min_points_calib:
    vec = np.random.rand(3)  # hier ersetzen mit median_od_vector_from_points
    st.session_state.stain_vec = blend_vectors(st.session_state.stain_vec, vec, weight_new=iter_blend)
    st.session_state.cal_points = []
    st.session_state.last_auto_run += 1
    st.success("✅ OD-Kalibrierung durchgeführt.")

# -------------------- Auto-Erkennung --------------------
if st.session_state.last_auto_run > 0:
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)
    mask_all = detect_all_nuclei(proc, st.session_state.stain_vec, min_area=min_area)
    detected_all = get_centers(mask_all, min_area)
    st.session_state.all_auto = dedup_points(apply_dbscan(detected_all, cluster_eps, cluster_min_samples),
                                             min_dist=max(4, circle_radius//2))
    st.session_state.last_auto_run = 0

# -------------------- Anzeige & Export --------------------
all_auto = st.session_state.get("all_auto", [])
all_manual = st.session_state.get("manual_points", [])

st.markdown("### 📊 Gesamtanzahl Zellkerne")
st.metric("Automatisch erkannt", len(all_auto))
st.metric("Manuell hinzugefügt", len(all_manual))
st.markdown(f"**Gesamtpunkte:** {len(all_auto)+len(all_manual)}")

# Ergebnisbild
result_img = image_disp.copy()
for (x, y) in all_auto:
    cv2.circle(result_img, (x, y), circle_radius, (0, 0, 255), 2)
for (x, y) in all_manual:
    cv2.circle(result_img, (x, y), circle_radius, (0, 165, 255), -1)

st.image(result_img, caption="Alle Zellkerne (auto = Outline, manuell = filled)", use_column_width=True)

# CSV Export
rows = [{"X_display": x, "Y_display": y, "Source": "auto"} for x,y in all_auto] + \
       [{"X_display": x, "Y_display": y, "Source": "manual"} for x,y in all_manual]

if rows:
    df = pd.DataFrame(rows)
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="zellkerne_gesamt.csv", mime="text/csv")
