# canvas2_auto_calib_fixed_deconv.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
from pathlib import Path
from sklearn.cluster import DBSCAN  # sicherstellen, dass importiert ist

# -------------------- Hilfsfunktionen (neu: OD & Deconv) --------------------

def od_from_rgb(img_rgb):
    """Berechnet optische Dichte (OD) aus einem RGB-Bild (uint8)."""
    # convert to float and avoid zeros
    imf = img_rgb.astype(np.float32)
    imf = np.clip(imf, 0, 255)
    od = -np.log((imf + 1.0) / 255.0)
    return od  # shape (H,W,3)

def sample_patch_pixels(img_rgb, x, y, radius=5):
    """Gebe alle RGB-Pixel im Displaybild innerhalb des Patch-Radius zurück."""
    h, w = img_rgb.shape[:2]
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)
    patch = img_rgb[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    return patch.reshape(-1, 3).astype(np.float32)

def median_od_vector_from_points(img_rgb, points, radius=5):
    """Ermittelt den medianen OD-Vektor (3,) aus einer Liste von Klickpunkten."""
    if not points:
        return None
    all_pixels = []
    for (x, y) in points:
        patch = sample_patch_pixels(img_rgb, x, y, radius)
        if patch.size > 0:
            all_pixels.append(patch)
    if not all_pixels:
        return None
    all_pixels = np.vstack(all_pixels)  # N x 3
    od = -np.log((all_pixels + 1.0) / 255.0)
    # robust: median pro Kanal
    med = np.median(od, axis=0)
    # if near-zero, return None
    if np.linalg.norm(med) < 1e-6:
        return None
    return med

def normalize_vector(v):
    v = v.astype(np.float32)
    n = np.linalg.norm(v)
    if n == 0 or np.isnan(n):
        return v
    return v / n

def make_stain_matrix(aec_vec, hema_vec, bg_vec=None):
    """
    Baut eine 3x3 'stain matrix' auf, Spalten sind die (normalisierten) OD-Vektoren.
    Falls nur zwei Vektoren vorhanden, erschaffen wir eine dritte orthogonale approximation (background).
    """
    # ensure arrays or fallbacks
    aec_v = normalize_vector(np.array(aec_vec)) if aec_vec is not None else None
    hema_v = normalize_vector(np.array(hema_vec)) if hema_vec is not None else None

    if aec_v is None and hema_v is None:
        return None

    cols = []
    if aec_v is not None:
        cols.append(aec_v)
    if hema_v is not None:
        # ensure not colinear
        if aec_v is not None and np.allclose(np.abs(np.dot(aec_v, hema_v)), 1.0, atol=1e-3):
            # slight jitter to avoid singularity
            hema_v = hema_v + 1e-3
            hema_v = normalize_vector(hema_v)
        cols.append(hema_v)

    # add background vector
    if bg_vec is not None:
        bg_v = normalize_vector(np.array(bg_vec))
        cols.append(bg_v)
    else:
        # create third vector orthogonal-ish via cross product if we have >=2
        if len(cols) >= 2:
            third = np.cross(cols[0], cols[1])
            if np.linalg.norm(third) < 1e-6:
                # fallback: use mean of remaining RGB OD
                third = np.array([0.01, 0.01, 0.01])
            third = normalize_vector(third)
            cols.append(third)
        else:
            # single vector case: pick two small orthogonal dims
            cols.append(normalize_vector(np.array([0.01,0.01,0.01])))

    M = np.column_stack(cols)  # 3x3
    # ensure invertible-ish by slight regularization if needed
    if np.linalg.matrix_rank(M) < 3:
        M = M + np.eye(3) * 1e-6
    return M

def compute_concentration_maps(od_img, stain_matrix):
    """
    od_img: HxWx3
    stain_matrix: 3x3 (columns: stain vectors)
    returns: list of HxW concentration maps (len == number of stains == 3)
    """
    if stain_matrix is None:
        h, w = od_img.shape[:2]
        return [np.zeros((h, w), dtype=np.float32)] * 3
    # solve concentrations: C = pinv(M) @ OD_pixel
    pinv = np.linalg.pinv(stain_matrix)  # 3x3
    h, w = od_img.shape[:2]
    od_flat = od_img.reshape(-1, 3).T  # 3 x N
    conc_flat = pinv @ od_flat  # 3 x N
    conc = conc_flat.T.reshape(h, w, 3)  # H W 3
    # clamp negatives to zero (concentrations should be >=0)
    conc = np.maximum(conc, 0.0)
    maps = [conc[..., i].astype(np.float32) for i in range(3)]
    return maps

# -------------------- Deine Funktionen wiederverwendet --------------------

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

def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, min_dist=6):
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    """Erweiterte Konturenerkennung mit Glättung und Morphologie."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    for c in contours:
        c = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        area = cv2.contourArea(c)
        if area >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0):
                cx = int(round(M["m10"] / M["m00"]))
                cy = int(round(M["m01"] / M["m00"]))
                centers.append((cx, cy))
    return centers

# -------------------- Save / Load Kalibrierung (angepasst) --------------------

def save_last_calibration(path="kalibrierung_deconv.json"):
    data = {
        "aec_vec": st.session_state.get("aec_vec").tolist() if st.session_state.get("aec_vec") is not None else None,
        "hema_vec": st.session_state.get("hema_vec").tolist() if st.session_state.get("hema_vec") is not None else None,
        "bg_vec": st.session_state.get("bg_vec").tolist() if st.session_state.get("bg_vec") is not None else None,
        "aec_threshold": float(st.session_state.get("aec_threshold", 0.0)),
        "hema_threshold": float(st.session_state.get("hema_threshold", 0.0))
    }
    try:
        with open(path, "w") as f:
            json.dump(data, f)
        st.success("💾 Kalibrierung (Deconv) gespeichert.")
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")

def load_last_calibration(path="kalibrierung_deconv.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        st.session_state.aec_vec = np.array(data.get("aec_vec")) if data.get("aec_vec") else None
        st.session_state.hema_vec = np.array(data.get("hema_vec")) if data.get("hema_vec") else None
        st.session_state.bg_vec = np.array(data.get("bg_vec")) if data.get("bg_vec") else None
        st.session_state.aec_threshold = float(data.get("aec_threshold", 0.0))
        st.session_state.hema_threshold = float(data.get("hema_threshold", 0.0))
        st.success("✅ Letzte Deconv-Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
def ensure_odd(k: int) -> int:
    """Sorgt dafür, dass Kernelgrößen ungerade sind (1,3,5,7...)."""
    if k < 1:
        return 1
    return k if k % 2 == 1 else k + 1

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (Deconv Auto-Kalib)", layout="wide")
st.title("🧬 Zellkern-Zähler – OD + Deconvolution (Auto-Kalib)")

# -------------------- Session State --------------------
default_lists = [
    "aec_cal_points", "hema_cal_points", "bg_cal_points",
    "aec_auto", "hema_auto",
    "manual_aec", "manual_hema",
    "aec_vec", "hema_vec", "bg_vec",
    "last_file", "disp_width", "last_auto_run",
    "aec_threshold", "hema_threshold"
]
for key in default_lists:
    if key not in st.session_state:
        if key in ["aec_vec", "hema_vec", "bg_vec"]:
            st.session_state[key] = None
        elif key == "disp_width":
            st.session_state[key] = 1400
        elif key in ["aec_threshold", "hema_threshold"]:
            st.session_state[key] = 0.0
        else:
            st.session_state[key] = []

# per-mode first-click-ignore flags
for flag in ["aec_first_ignore", "hema_first_ignore", "bg_first_ignore"]:
    if flag not in st.session_state:
        st.session_state[flag] = True

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset state on new file name
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "aec_auto", "hema_auto", "manual_aec", "manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_vec", "hema_vec", "bg_vec"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0
    st.session_state.aec_threshold = 0.0
    st.session_state.hema_threshold = 0.0

# -------------------- Bild vorbereiten --------------------
colW1, colW2 = st.columns([2, 1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)

# OD image (display scale) used for calibration and deconv
od_disp = od_from_rgb(image_disp)  # H W 3

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Deconvolution & Kalibrierung")
blur_kernel = max(1, int(st.sidebar.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5, step=1)))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 500, 50)

st.sidebar.markdown("### 🧩 DBSCAN-Clustering für Auto-Punkte")
cluster_eps = st.sidebar.number_input("Cluster-Radius (eps)", 1, 500, 25)
cluster_min_samples = st.sidebar.number_input("Min. Punkte pro Cluster", 1, 20, 1)

alpha = st.sidebar.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)
calib_patch_radius = st.sidebar.slider("🎯 OD-Patch-Radius (px)", 1, 20, 5)
min_points_calib = st.sidebar.slider("🧮 Minimale Punkte für automatische Kalibrierung", 1, 10, 3, step=1)
st.sidebar.info("Kalibrierung läuft automatisch, sobald die minimale Punktzahl erreicht ist.")

# Deconv-specific params
st.sidebar.markdown("### 🔬 Deconv Einstellungen")
aec_thresh_mul = st.sidebar.slider("AEC Threshold-Multiplikator", 0.1, 5.0, 1.0, step=0.1)
hema_thresh_mul = st.sidebar.slider("Hämatoxylin Threshold-Multiplikator", 0.1, 5.0, 1.0, step=0.1)
iter_blend = st.sidebar.slider("Iterative Kalibrierung: Blend-Gewicht (neu vs. alt)", 0.0, 1.0, 0.6, step=0.05)

if st.sidebar.button("🔁 Kalibrierung laden"):
    load_last_calibration()

if st.sidebar.button("💾 Kalibrierung speichern"):
    save_last_calibration()

# Modes
MODES = [
    "AEC Kalibrier-Punkt setzen",
    "Hämatoxylin Kalibrier-Punkt setzen",
    "Hintergrund Kalibrier-Punkt setzen",
    "AEC manuell hinzufügen",
    "Hämatoxylin manuell hinzufügen",
    "Punkt löschen"
]
mode = st.sidebar.radio("Modus", MODES, index=0)

aec_mode = mode == MODES[0]
hema_mode = mode == MODES[1]
bg_mode = mode == MODES[2]
manual_aec_mode = mode == MODES[3]
manual_hema_mode = mode == MODES[4]
delete_mode = mode == MODES[5]

if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = None

if mode != st.session_state.prev_mode:
    if aec_mode:
        st.session_state.aec_first_ignore = True
    if hema_mode:
        st.session_state.hema_first_ignore = True
    if bg_mode:
        st.session_state.bg_first_ignore = True
    st.session_state.prev_mode = mode

if st.sidebar.button("🧹 Alle Punkte löschen"):
    for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "aec_auto", "hema_auto", "manual_aec", "manual_hema"]:
        st.session_state[k] = []
    st.success("Alle Punkte gelöscht.")

# -------------------- Bildanzeige mit Markierungen --------------------
marked_disp = image_disp.copy()
for (x, y) in st.session_state.aec_cal_points:
    cv2.circle(marked_disp, (x, y), max(2, circle_radius), (0, 120, 200), -1)
for (x, y) in st.session_state.hema_cal_points:
    cv2.circle(marked_disp, (x, y), max(2, circle_radius), (200, 120, 0), -1)
for (x, y) in st.session_state.bg_cal_points:
    cv2.circle(marked_disp, (x, y), max(2, circle_radius), (200, 200, 0), -1)
for (x, y) in st.session_state.manual_aec:
    cv2.circle(marked_disp, (x, y), circle_radius, (0, 165, 255), -1)
for (x, y) in st.session_state.manual_hema:
    cv2.circle(marked_disp, (x, y), circle_radius, (128, 0, 128), -1)
for (x, y) in st.session_state.aec_auto:
    cv2.circle(marked_disp, (x, y), circle_radius, (0, 0, 255), 2)
for (x, y) in st.session_state.hema_auto:
    cv2.circle(marked_disp, (x, y), circle_radius, (255, 0, 0), 2)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key=f"clickable_image_{st.session_state.last_file}", width=DISPLAY_WIDTH)

# -------------------- Klicklogik --------------------
if coords:
    x, y = int(coords["x"]), int(coords["y"])

    if delete_mode:
        for key in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "manual_aec", "manual_hema"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p, (x, y), circle_radius)]
        st.info("Punkt(e) gelöscht (falls gefunden).")

    elif aec_mode:
        if st.session_state.aec_first_ignore:
            st.session_state.aec_first_ignore = False
            st.info("⏳ Erster AEC-Klick ignoriert (Initialisierung).")
        else:
            st.session_state.aec_cal_points.append((x, y))
            st.info(f"📍 AEC-Kalibrierpunkt hinzugefügt ({x}, {y})")

    elif hema_mode:
        if st.session_state.hema_first_ignore:
            st.session_state.hema_first_ignore = False
            st.info("⏳ Erster Hämatoxylin-Klick ignoriert (Initialisierung).")
        else:
            st.session_state.hema_cal_points.append((x, y))
            st.info(f"📍 Hämatoxylin-Kalibrierpunkt hinzugefügt ({x}, {y})")

    elif bg_mode:
        if st.session_state.bg_first_ignore:
            st.session_state.bg_first_ignore = False
            st.info("⏳ Erster Hintergrund-Klick ignoriert (Initialisierung).")
        else:
            st.session_state.bg_cal_points.append((x, y))
            st.info(f"📍 Hintergrund-Kalibrierpunkt hinzugefügt ({x}, {y})")

    elif manual_aec_mode:
        st.session_state.manual_aec.append((x, y))
        st.info(f"✋ Manuell: AEC-Punkt ({x}, {y})")

    elif manual_hema_mode:
        st.session_state.manual_hema.append((x, y))
        st.info(f"✋ Manuell: Hämatoxylin-Punkt ({x}, {y})")

# Deduplication (einmal)
for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "manual_aec", "manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius // 2))

# -------------------- Auto-Kalibrierung (OD/Deconv) --------------------
calibrated_any = False

# helper: update/merge vectors iteratively
def blend_vectors(old, new, weight_new=0.6):
    if old is None:
        return new
    if new is None:
        return old
    return normalize_vector((1.0 - weight_new) * old + weight_new * new)

if len(st.session_state.bg_cal_points) >= min_points_calib:
    vec_bg = median_od_vector_from_points(image_disp, st.session_state.bg_cal_points, radius=calib_patch_radius)
    if vec_bg is not None:
        st.session_state.bg_vec = blend_vectors(st.session_state.bg_vec, vec_bg, weight_new=iter_blend)
        st.session_state.bg_cal_points = []
        calibrated_any = True

if len(st.session_state.aec_cal_points) >= min_points_calib:
    vec_aec = median_od_vector_from_points(image_disp, st.session_state.aec_cal_points, radius=calib_patch_radius)
    if vec_aec is not None:
        st.session_state.aec_vec = blend_vectors(st.session_state.aec_vec, vec_aec, weight_new=iter_blend)
        st.session_state.aec_cal_points = []
        calibrated_any = True

if len(st.session_state.hema_cal_points) >= min_points_calib:
    vec_hema = median_od_vector_from_points(image_disp, st.session_state.hema_cal_points, radius=calib_patch_radius)
    if vec_hema is not None:
        st.session_state.hema_vec = blend_vectors(st.session_state.hema_vec, vec_hema, weight_new=iter_blend)
        st.session_state.hema_cal_points = []
        calibrated_any = True

if calibrated_any:
    st.session_state.last_auto_run += 1
    st.success("✅ Deconv-Auto-Kalibrierung durchgeführt.")

# -------------------- Auto-Erkennung (Deconv) --------------------
if st.session_state.last_auto_run > 0:
    # vorbereitete Bildverarbeitung (Kontrast + optional Blur)
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)

    od_proc = od_from_rgb(proc)  # H W 3

    # Stain-Matrix bauen
    M = make_stain_matrix(st.session_state.aec_vec, st.session_state.hema_vec, st.session_state.bg_vec)

    # Konzentrationsmaps
    conc_maps = compute_concentration_maps(od_proc, M)  # list of 3 maps: [aec_map, hema_map, bg_map-or-other]
    aec_map = conc_maps[0]
    hema_map = conc_maps[1]
    # bg_map = conc_maps[2]  # optional

    # thresholds automatisch bestimmen (robust median)
    # avoid zero-median by small epsilon
    aec_med = float(np.median(aec_map[aec_map > 0])) if np.any(aec_map > 0) else float(np.median(aec_map))
    hema_med = float(np.median(hema_map[hema_map > 0])) if np.any(hema_map > 0) else float(np.median(hema_map))

    if aec_med <= 0:
        aec_med = np.mean(aec_map) + 1e-6
    if hema_med <= 0:
        hema_med = np.mean(hema_map) + 1e-6

    # store thresholds in session (iterative calibration friendly)
    st.session_state.aec_threshold = aec_med * aec_thresh_mul
    st.session_state.hema_threshold = hema_med * hema_thresh_mul

    # create masks
    mask_aec = (aec_map >= st.session_state.aec_threshold).astype(np.uint8) * 255
    mask_hema = (hema_map >= st.session_state.hema_threshold).astype(np.uint8) * 255

    # remove background via bg_vec if available (optional)
    if st.session_state.bg_vec is not None:
        bg_map = conc_maps[2]
        mask_bg = (bg_map >= (np.median(bg_map[bg_map > 0]) if np.any(bg_map > 0) else 0)).astype(np.uint8) * 255
        mask_aec = cv2.bitwise_and(mask_aec, cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema, cv2.bitwise_not(mask_bg))

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel)
    mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, kernel)

    # Centers erkennen (wie vorher)
    detected_aec = get_centers(mask_aec, int(min_area))
    detected_hema = get_centers(mask_hema, int(min_area))

    # DBSCAN-Clustering anwenden
    clustered_aec = apply_dbscan(detected_aec, cluster_eps, cluster_min_samples)
    clustered_hema = apply_dbscan(detected_hema, cluster_eps, cluster_min_samples)

    # Deduplication nach Cluster
    st.session_state.aec_auto = dedup_points(clustered_aec, min_dist=max(4, circle_radius // 2))
    st.session_state.hema_auto = dedup_points(clustered_hema, min_dist=max(4, circle_radius // 2))

    # Trigger zurücksetzen
    st.session_state.last_auto_run = 0

# -------------------- Ergebnisse + Export --------------------
aec_auto = st.session_state.aec_auto or []
aec_manual = st.session_state.manual_aec or []
hema_auto = st.session_state.hema_auto or []
hema_manual = st.session_state.manual_hema or []

st.markdown("### 📊 Ergebnisse")
colA, colB = st.columns(2)
with colA:
    st.metric("AEC (auto)", len(aec_auto))
    st.metric("AEC (manuell)", len(aec_manual))
with colB:
    st.metric("Hämatoxylin (auto)", len(hema_auto))
    st.metric("Hämatoxylin (manuell)", len(hema_manual))

# result image
result_img = image_disp.copy()
for (x, y) in aec_auto:
    cv2.circle(result_img, (x, y), circle_radius, (0, 0, 255), 2)
for (x, y) in hema_auto:
    cv2.circle(result_img, (x, y), circle_radius, (255, 0, 0), 2)
for (x, y) in aec_manual:
    cv2.circle(result_img, (x, y), circle_radius, (0, 165, 255), -1)
for (x, y) in hema_manual:
    cv2.circle(result_img, (x, y), circle_radius, (128, 0, 128), -1)

if isinstance(result_img, np.ndarray):
    if result_img.dtype != np.uint8:
        result_img = np.clip(result_img, 0, 255).astype(np.uint8)
    try:
        st.image(result_img, caption="Erkannte Punkte (auto = outline, manuell = filled)", use_column_width=True)
    except TypeError:
        st.image(result_img, caption="Erkannte Punkte (auto = outline, manuell = filled)")

# CSV export
rows = []
for x, y in aec_auto:
    rows.append({"X_display": x, "Y_display": y, "Type": "AEC", "Source": "auto"})
for x, y in aec_manual:
    rows.append({"X_display": x, "Y_display": y, "Type": "AEC", "Source": "manual"})
for x, y in hema_auto:
    rows.append({"X_display": x, "Y_display": y, "Type": "Hämatoxylin", "Source": "auto"})
for x, y in hema_manual:
    rows.append({"X_display": x, "Y_display": y, "Type": "Hämatoxylin", "Source": "manual"})

if rows:
    df = pd.DataFrame(rows)
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren", data=df.to_csv(index=False).encode("utf-8"), file_name="zellkerne_deconv.csv", mime="text/csv")

# Debug
with st.expander("🧠 Debug Info"):
    st.write({
        "aec_vec": (st.session_state.aec_vec.tolist() if st.session_state.aec_vec is not None else None),
        "hema_vec": (st.session_state.hema_vec.tolist() if st.session_state.hema_vec is not None else None),
        "bg_vec": (st.session_state.bg_vec.tolist() if st.session_state.bg_vec is not None else None),
        "aec_threshold": st.session_state.aec_threshold,
        "hema_threshold": st.session_state.hema_threshold,
        "aec_auto_count": len(st.session_state.aec_auto),
        "hema_auto_count": len(st.session_state.hema_auto),
        "manual_aec_count": len(st.session_state.manual_aec),
        "manual_hema_count": len(st.session_state.manual_hema),
        "aec_cal_points": st.session_state.aec_cal_points,
        "hema_cal_points": st.session_state.hema_cal_points,
        "bg_cal_points": st.session_state.bg_cal_points,
        "last_auto_run": st.session_state.last_auto_run,
        "image_shape": image_disp.shape if isinstance(image_disp, np.ndarray) else None
    })
