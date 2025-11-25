# -------------------- canvas2_auto_calib_od.py --------------------
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from pathlib import Path
import json
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

def get_centers(mask, min_area=50):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        c = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True), True)
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0):
                cx = int(round(M["m10"]/M["m00"]))
                cy = int(round(M["m01"]/M["m00"]))
                centers.append((cx, cy))
    return centers

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

# -------------------- OD / Deconv Helper --------------------
def extract_patch(img, x, y, radius=4):
    H, W = img.shape[:2]
    x_min = max(0, x-radius)
    x_max = min(W, x+radius+1)
    y_min = max(0, y-radius)
    y_max = min(H, y+radius+1)
    if x_min >= x_max or y_min >= y_max:
        return None
    return img[y_min:y_max, x_min:x_max]

def rgb_to_od(patch, eps=1e-6):
    patch = patch.astype(np.float32)
    return -np.log((patch + eps)/255.0)

def normalize_vector(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    return v/norm if norm>0 else v

def median_od_vector_from_points(img, points, radius=4):
    vals = []
    for (x, y) in points:
        patch = extract_patch(img, x, y, radius)
        if patch is None or patch.size==0:
            continue
        od_patch = rgb_to_od(patch)
        v = np.median(od_patch.reshape(-1,3), axis=0)
        if np.linalg.norm(v)==0 or np.any(np.isnan(v)):
            continue
        vals.append(v)
    if len(vals)==0:
        return None
    return normalize_vector(np.median(vals, axis=0))

def blend_vectors(old, new, weight_new=0.6):
    if old is None and new is None:
        return None
    if old is None:
        return normalize_vector(new)
    if new is None:
        return old
    mixed = (1-weight_new)*old + weight_new*new
    return normalize_vector(mixed)

def make_stain_matrix(aec_vec, hema_vec, bg_vec):
    aec = normalize_vector(aec_vec)
    hema = normalize_vector(hema_vec)
    bg = normalize_vector(bg_vec)
    M = np.stack([aec, hema, bg], axis=1) + np.eye(3)*1e-6
    return M

def deconvolve_image(img, M):
    H, W = img.shape[:2]
    OD = -np.log((img.astype(np.float32)+1e-6)/255.0).reshape(-1,3)
    try:
        C = np.linalg.lstsq(M, OD.T, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return C.T.reshape(H,W,3)

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (OD/Deconv)", layout="wide")
st.title("🧬 Zellkern-Zähler – OD/Deconv Auto-Kalibrierung")

# -------------------- Session State --------------------
keys = ["aec_cal_points","hema_cal_points","bg_cal_points","manual_aec","manual_hema",
        "aec_auto","hema_auto","aec_vec","hema_vec","bg_vec","last_file","disp_width","last_auto_run"]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = [] if 'points' in k or 'auto' in k else None
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = None
if "first_ignore_flags" not in st.session_state:
    st.session_state.first_ignore_flags = {"aec": True,"hema": True,"bg": True}

# -------------------- Upload & Display --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.last_file = uploaded_file.name
    for k in ["aec_cal_points","hema_cal_points","bg_cal_points","manual_aec","manual_hema","aec_auto","hema_auto"]:
        st.session_state[k] = []
    st.session_state.aec_vec = st.session_state.hema_vec = st.session_state.bg_vec = None
    st.session_state.last_auto_run = 0

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400,2000,1400,step=100)
scale = DISPLAY_WIDTH/W_orig
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H_orig*scale)), interpolation=cv2.INTER_AREA)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Kalibrierung & Filter")
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)",1,20,5)
calib_patch_radius = st.sidebar.slider("🎯 Kalibrierungsradius (px)",1,15,4)
min_points_calib = st.sidebar.slider("🧮 Minimale Punkte für Kalibrierung",1,10,3)
iter_blend = st.sidebar.slider("🔄 Gewicht neues Patch",0.1,0.9,0.6,0.05)
min_area = st.sidebar.number_input("📏 Mindestfläche (px)",10,500,50)
cluster_eps = st.sidebar.number_input("Cluster-Radius (eps)",1,500,25)
cluster_min_samples = st.sidebar.number_input("Min. Punkte pro Cluster",1,20,1)

MODES = ["AEC Kalibrier-Punkt setzen","Hämatoxylin Kalibrier-Punkt setzen","Hintergrund Kalibrier-Punkt setzen",
         "AEC manuell hinzufügen","Hämatoxylin manuell hinzufügen","Punkt löschen"]
mode = st.sidebar.radio("Modus",MODES,0)
aec_mode = mode==MODES[0]
hema_mode = mode==MODES[1]
bg_mode = mode==MODES[2]
manual_aec_mode = mode==MODES[3]
manual_hema_mode = mode==MODES[4]
delete_mode = mode==MODES[5]

# -------------------- Klicklogik & iterative OD-Kalib --------------------
if coords:
    x, y = int(coords["x"]), int(coords["y"])

    # Punkte löschen
    if delete_mode:
        for key in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "manual_aec", "manual_hema"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p, (x, y), circle_radius)]
        st.info("Punkt(e) gelöscht (falls gefunden).")

    # AEC-Kalib
    elif aec_mode:
        if st.session_state.aec_first_ignore:
            st.session_state.aec_first_ignore = False
            st.info("⏳ Erster AEC-Klick ignoriert (Initialisierung).")
        else:
            st.session_state.aec_cal_points.append((x, y))
            st.info(f"📍 AEC-Kalibrierpunkt hinzugefügt ({x}, {y})")

    # Hämatoxylin-Kalib
    elif hema_mode:
        if st.session_state.hema_first_ignore:
            st.session_state.hema_first_ignore = False
            st.info("⏳ Erster Hämatoxylin-Klick ignoriert (Initialisierung).")
        else:
            st.session_state.hema_cal_points.append((x, y))
            st.info(f"📍 Hämatoxylin-Kalibrierpunkt hinzugefügt ({x}, {y})")

    # Hintergrund-Kalib
    elif bg_mode:
        if st.session_state.bg_first_ignore:
            st.session_state.bg_first_ignore = False
            st.info("⏳ Erster Hintergrund-Klick ignoriert (Initialisierung).")
        else:
            st.session_state.bg_cal_points.append((x, y))
            st.info(f"📍 Hintergrund-Kalibrierpunkt hinzugefügt ({x}, {y})")

    # Manuelle Punkte
    elif manual_aec_mode:
        st.session_state.manual_aec.append((x, y))
        st.info(f"✋ Manuell: AEC-Punkt ({x}, {y})")

    elif manual_hema_mode:
        st.session_state.manual_hema.append((x, y))
        st.info(f"✋ Manuell: Hämatoxylin-Punkt ({x}, {y})")

# Deduplication
for k in ["aec_cal_points", "hema_cal_points", "bg_cal_points", "manual_aec", "manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius // 2))

# -------------------- Iterative OD Auto-Kalibrierung --------------------
calibrated_any = False

def blend_vectors(old, new, weight_new=0.6):
    if old is None:
        return new
    if new is None:
        return old
    return normalize_vector((1.0 - weight_new) * old + weight_new * new)

# Calib-Patches berechnen & iterativ aktualisieren
for color, key_points, key_vec in [
    ("bg", st.session_state.bg_cal_points, "bg_vec"),
    ("aec", st.session_state.aec_cal_points, "aec_vec"),
    ("hema", st.session_state.hema_cal_points, "hema_vec")
]:
    if len(key_points) >= min_points_calib:
        vec = median_od_vector_from_points(image_disp, key_points, radius=calib_patch_radius)
        if vec is not None:
            st.session_state[key_vec] = blend_vectors(st.session_state.get(key_vec), vec, weight_new=iter_blend)
            st.session_state[key_points] = []
            calibrated_any = True

if calibrated_any:
    st.session_state.last_auto_run += 1
    st.success("✅ Iterative Deconv-Kalibrierung durchgeführt.")

# -------------------- Ergebnis + Live Counts + CSV --------------------
aec_auto = st.session_state.aec_auto or []
aec_manual = st.session_state.manual_aec or []
hema_auto = st.session_state.hema_auto or []
hema_manual = st.session_state.manual_hema or []

st.markdown("### 📊 Anzahl erkannter Punkte")
colA, colB = st.columns(2)
with colA:
    st.metric("AEC (auto)", len(aec_auto))
    st.metric("AEC (manuell)", len(aec_manual))
with colB:
    st.metric("Hämatoxylin (auto)", len(hema_auto))
    st.metric("Hämatoxylin (manuell)", len(hema_manual))

st.markdown(f"**Gesamtpunkte:** {len(aec_auto)+len(aec_manual)+len(hema_auto)+len(hema_manual)}")

# Ergebnisbild
result_img = image_disp.copy()
for (x, y) in aec_auto:
    cv2.circle(result_img, (x, y), circle_radius, (0, 0, 255), 2)
for (x, y) in hema_auto:
    cv2.circle(result_img, (x, y), circle_radius, (255, 0, 0), 2)
for (x, y) in aec_manual:
    cv2.circle(result_img, (x, y), circle_radius, (0, 165, 255), -1)
for (x, y) in hema_manual:
    cv2.circle(result_img, (x, y), circle_radius, (128, 0, 128), -1)

st.image(result_img, caption="Erkannte Punkte (auto = Outline, manuell = filled)", use_column_width=True)

# CSV Export
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
    st.download_button(
        "📥 CSV exportieren",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="zellkerne_iterative.csv",
        mime="text/csv"
    )

# -------------------- Auto-Erkennung --------------------
if all(v is not None for v in [st.session_state.aec_vec, st.session_state.hema_vec, st.session_state.bg_vec]):
    if st.session_state.last_auto_run == 0:
        try:
            M = make_stain_matrix(st.session_state.aec_vec, st.session_state.hema_vec, st.session_state.bg_vec)
        except:
            M = None

        if M is not None:
            C = deconvolve_image(image_disp, M)
            if C is not None:
                bg_od = C[:,:,2]
                mask_bg = bg_od < np.median(bg_od)*1.2
                mask_aec = (C[:,:,0] > np.median(C[:,:,0][mask_bg])+0.05) & mask_bg
                mask_hema = (C[:,:,1] > np.median(C[:,:,1][mask_bg])+0.05) & mask_bg
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                mask_aec = cv2.morphologyEx((mask_aec.astype(np.uint8)*255),cv2.MORPH_OPEN,kernel)
                mask_hema = cv2.morphologyEx((mask_hema.astype(np.uint8)*255),cv2.MORPH_OPEN,kernel)

                detected_aec = get_centers(mask_aec,min_area=int(min_area))
                detected_hema = get_centers(mask_hema,min_area=int(min_area))
                st.session_state.aec_auto = dedup_points(apply_dbscan(detected_aec,cluster_eps,cluster_min_samples),
                                                         min_dist=max(4,circle_radius//2))
                st.session_state.hema_auto = dedup_points(apply_dbscan(detected_hema,cluster_eps,cluster_min_samples),
                                                          min_dist=max(4,circle_radius//2))
                st.session_state.last_auto_run = 1

# -------------------- Ergebnisse + Export (mit Live-Zahlen) --------------------
aec_auto = st.session_state.aec_auto or []
aec_manual = st.session_state.manual_aec or []
hema_auto = st.session_state.hema_auto or []
hema_manual = st.session_state.manual_hema or []

st.markdown("### 📊 Anzahl erkannter Punkte")
colA, colB = st.columns(2)
with colA:
    st.metric("AEC (auto)", len(aec_auto))
    st.metric("AEC (manuell)", len(aec_manual))
with colB:
    st.metric("Hämatoxylin (auto)", len(hema_auto))
    st.metric("Hämatoxylin (manuell)", len(hema_manual))

st.markdown(f"**Gesamtpunkte:** {len(aec_auto)+len(aec_manual)+len(hema_auto)+len(hema_manual)}")

# Ergebnisbild mit allen Markierungen
result_img = image_disp.copy()
for (x, y) in aec_auto:
    cv2.circle(result_img, (x, y), circle_radius, (0, 0, 255), 2)
for (x, y) in hema_auto:
    cv2.circle(result_img, (x, y), circle_radius, (255, 0, 0), 2)
for (x, y) in aec_manual:
    cv2.circle(result_img, (x, y), circle_radius, (0, 165, 255), -1)
for (x, y) in hema_manual:
    cv2.circle(result_img, (x, y), circle_radius, (128, 0, 128), -1)

st.image(result_img, caption="Erkannte Punkte (auto = Outline, manuell = filled)", use_column_width=True)

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
    st.download_button(
        "📥 CSV exportieren",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="zellkerne_v5.csv",
        mime="text/csv"
    )
