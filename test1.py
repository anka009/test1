# canvas2_auto_calib_fixed.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
from pathlib import Path
from sklearn.cluster import DBSCAN  # sicherstellen, dass importiert ist

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

# -------------------- Hilfsfunktionen --------------------

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


def compute_hsv_range(points, hsv_img, radius=5):
    """Robuste Median-basierte HSV-Range-Berechnung mit Wrap-Achtung für Hue."""
    if not points:
        return None
    vals = []
    for (x, y) in points:
        x_min = max(0, x - radius)
        x_max = min(hsv_img.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(hsv_img.shape[0], y + radius + 1)
        region = hsv_img[y_min:y_max, x_min:x_max]
        if region.size > 0:
            vals.append(region.reshape(-1, 3))
    if not vals:
        return None
    vals = np.vstack(vals)
    h = vals[:, 0].astype(int)
    s = vals[:, 1].astype(int)
    v = vals[:, 2].astype(int)

    h_med = float(np.median(h))
    s_med = float(np.median(s))
    v_med = float(np.median(v))

    n_points = len(points)
    tol_h = int(min(25, 10 + n_points * 3))
    tol_s = int(min(80, 30 + n_points * 10))
    tol_v = int(min(80, 30 + n_points * 10))

    # Hue wrap aware
    if np.mean(h) > 150 or np.mean(h) < 20:
        h_adjusted = np.where(h < 90, h + 180, h)
        h_med = float(np.median(h_adjusted) % 180)
        tol_h = min(40, tol_h + 5)

    h_min = int(round((h_med - tol_h) % 180))
    h_max = int(round((h_med + tol_h) % 180))
    s_min = max(0, int(round(s_med - tol_s)))
    s_max = min(255, int(round(s_med + tol_s)))
    v_min = max(0, int(round(v_med - tol_v)))
    v_max = min(255, int(round(v_med + tol_v)))

    return (h_min, h_max, s_min, s_max, v_min, v_max)


def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    # ensure ints and in valid ranges
    hmin, hmax, smin, smax, vmin, vmax = map(int, (hmin, hmax, smin, smax, vmin, vmax))
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([179, smax, vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask


def ensure_odd(k):
    return k if k % 2 == 1 else k + 1


def remove_near(points, forbidden_points, r):
    if not forbidden_points:
        return points
    return [p for p in points if not any(is_near(p, q, r) for q in forbidden_points)]


def save_last_calibration(path="kalibrierung.json"):
    def safe_list(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, list):
            return val
        else:
            return None
    data = {
        "aec_hsv": safe_list(st.session_state.get("aec_hsv")),
        "hema_hsv": safe_list(st.session_state.get("hema_hsv")),
        "bg_hsv": safe_list(st.session_state.get("bg_hsv"))
    }
    try:
        with open(path, "w") as f:
            json.dump(data, f)
        st.success("💾 Kalibrierung gespeichert.")
    except Exception as e:
        st.error(f"Fehler beim Speichern: {e}")


def load_last_calibration(path="kalibrierung.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        st.session_state.aec_hsv = np.array(data.get("aec_hsv")) if data.get("aec_hsv") else None
        st.session_state.hema_hsv = np.array(data.get("hema_hsv")) if data.get("hema_hsv") else None
        st.session_state.bg_hsv = np.array(data.get("bg_hsv")) if data.get("bg_hsv") else None
        st.success("✅ Letzte Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (Auto-Kalib) - Fixed", layout="wide")
st.title("🧬 Zellkern-Zähler – Auto-Kalibrierung (AEC / Hämatoxylin) — korrigiert")

# -------------------- Session State --------------------
default_lists = [
    "aec_cal_points", "hema_cal_points", "bg_cal_points",
    "aec_auto", "hema_auto",
    "manual_aec", "manual_hema",
    "aec_hsv", "hema_hsv", "bg_hsv",
    "last_file", "disp_width", "last_auto_run"
]
for key in default_lists:
    if key not in st.session_state:
        if key in ["aec_hsv", "hema_hsv", "bg_hsv"]:
            st.session_state[key] = None
        elif key == "disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = []

# per-mode first-click-ignore flags (only these — no global "first_click_ignored")
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
    for k in ["aec_hsv", "hema_hsv", "bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
colW1, colW2 = st.columns([2, 1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5, step=1))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 10, 2000, 100)
# -------------------- DBSCAN-Clustering für Auto-Punkte --------------------
st.sidebar.markdown("### 🧩 DBSCAN-Clustering für Auto-Punkte")
cluster_eps = st.sidebar.number_input("Cluster-Radius (eps)", 1, 200, 25)
cluster_min_samples = st.sidebar.number_input("Min. Punkte pro Cluster", 1, 20, 2)

alpha = st.sidebar.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius (Pixel)", 1, 15, 5)

min_points_calib = st.sidebar.slider(
    "🧮 Minimale Punkte für automatische Kalibrierung",
    min_value=1, max_value=10, value=3, step=1
)
st.sidebar.info("Kalibrierung läuft automatisch, sobald die minimale Punktzahl erreicht ist.")

# Modes — exact strings to avoid substring matching
MODES = [
    "AEC Kalibrier-Punkt setzen",
    "Hämatoxylin Kalibrier-Punkt setzen",
    "Hintergrund Kalibrier-Punkt setzen",
    "AEC manuell hinzufügen",
    "Hämatoxylin manuell hinzufügen",
    "Punkt löschen"
]
mode = st.sidebar.radio("Modus", MODES, index=0)

# boolean flags
aec_mode = mode == MODES[0]
hema_mode = mode == MODES[1]
bg_mode = mode == MODES[2]
manual_aec_mode = mode == MODES[3]
manual_hema_mode = mode == MODES[4]
delete_mode = mode == MODES[5]

# When switching mode, re-enable the respective ignore-flag (only exact matches)
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

# Quick actions
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

# KEY: do NOT include last_auto_run in the widget key anymore
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

# Deduplication (once)
for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "manual_aec", "manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius // 2))

# -------------------- Auto-Kalibrierung (sammelt Änderungen und führt genau 1x aus) --------------------
calibrated_any = False
# compute calibration ranges from display HSV (points are in display coords)
if len(st.session_state.bg_cal_points) >= min_points_calib:
    hsv_bg = compute_hsv_range(st.session_state.bg_cal_points, hsv_disp, radius=calib_radius)
    if hsv_bg is not None:
        st.session_state.bg_hsv = hsv_bg
        st.session_state.bg_cal_points = []
        calibrated_any = True

if len(st.session_state.aec_cal_points) >= min_points_calib:
    hsv_aec = compute_hsv_range(st.session_state.aec_cal_points, hsv_disp, radius=calib_radius)
    if hsv_aec is not None:
        st.session_state.aec_hsv = hsv_aec
        st.session_state.aec_cal_points = []
        calibrated_any = True

if len(st.session_state.hema_cal_points) >= min_points_calib:
    hsv_hema = compute_hsv_range(st.session_state.hema_cal_points, hsv_disp, radius=calib_radius)
    if hsv_hema is not None:
        st.session_state.hema_hsv = hsv_hema
        st.session_state.hema_cal_points = []
        calibrated_any = True

if calibrated_any:
    st.session_state.last_auto_run += 1
    st.success("✅ Auto-Kalibrierung durchgeführt.")

# -------------------- Auto-Erkennung (wenn letzte Kalibrierung erfolgte) --------------------
if st.session_state.last_auto_run > 0:
    # vorbereitete Bildverarbeitung (Kontrast + Blur)
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    # Maske erstellen
    mask_aec = apply_hue_wrap(hsv_proc, *st.session_state.aec_hsv) if st.session_state.aec_hsv else np.zeros(hsv_proc.shape[:2], dtype=np.uint8)
    mask_hema = apply_hue_wrap(hsv_proc, *st.session_state.hema_hsv) if st.session_state.hema_hsv else np.zeros(hsv_proc.shape[:2], dtype=np.uint8)
    if st.session_state.bg_hsv:
        mask_bg = apply_hue_wrap(hsv_proc, *st.session_state.bg_hsv)
        mask_aec = cv2.bitwise_and(mask_aec, cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema, cv2.bitwise_not(mask_bg))

    # Morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel)
    mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, kernel)

    # Centers erkennen
    detected_aec = get_centers(mask_aec, int(min_area))
    detected_hema = get_centers(mask_hema, int(min_area))

    # DBSCAN-Clustering anwenden
    clustered_aec = apply_dbscan(detected_aec, cluster_eps, cluster_min_samples)
    clustered_hema = apply_dbscan(detected_hema, cluster_eps, cluster_min_samples)

    # Deduplication nach Cluster
    st.session_state.aec_auto = dedup_points(clustered_aec, min_dist=max(4, circle_radius // 2))
    st.session_state.hema_auto = dedup_points(clustered_hema, min_dist=max(4, circle_radius // 2))

    # Trigger zurücksetzen, damit Auto-Erkennung nicht mehrfach läuft
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
    st.download_button("📥 CSV exportieren", data=df.to_csv(index=False).encode("utf-8"), file_name="zellkerne_v4_fixed.csv", mime="text/csv")

# Debug
with st.expander("🧠 Debug Info"):
    st.write({
        "aec_hsv": st.session_state.aec_hsv,
        "hema_hsv": st.session_state.hema_hsv,
        "bg_hsv": st.session_state.bg_hsv,
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

