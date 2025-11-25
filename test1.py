# canvas2_auto_calib_od.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
from pathlib import Path

# -------------------- Hilfsfunktionen --------------------
def normalize_vector(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)

def make_stain_matrix(aec_vec, hema_vec, bg_vec=None):
    aec_v = normalize_vector(aec_vec)
    hema_v = normalize_vector(hema_vec)
    if bg_vec is None:
        bg_v = np.cross(aec_v, hema_v)
        bg_v = normalize_vector(bg_v)
    else:
        bg_v = normalize_vector(bg_vec)
    M = np.stack([aec_v, hema_v, bg_v], axis=1)
    M += np.eye(3)*1e-6
    return M

def deconvolve(img, stain_matrix):
    od = -np.log((img.astype(float)+1)/255)
    deconv = np.linalg.lstsq(stain_matrix, od.reshape(-1,3).T, rcond=None)[0]
    return deconv.T.reshape(img.shape)

def extract_patch(img, x, y, radius=5):
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    return img[y_min:y_max, x_min:x_max]

def median_od_vector_from_points(img, points, radius=5):
    vectors = []
    for (x, y) in points:
        patch = extract_patch(img, x, y, radius)
        od_patch = -np.log((patch.astype(float)+1)/255)
        median_vec = np.median(od_patch.reshape(-1,3), axis=0)
        if np.linalg.norm(median_vec) > 0:
            vectors.append(normalize_vector(median_vec))
    if vectors:
        return np.median(np.stack(vectors, axis=0), axis=0)
    return None

def get_centers(mask, min_area=10):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx, cy))
    return centers

def iterative_calib(img_disp, aec_point=None, hema_point=None, bg_point=None, min_area=10):
    all_points = []
    vectors = {}
    if aec_point:
        vec_aec = median_od_vector_from_points(img_disp, [aec_point])
        vectors["aec"] = vec_aec
    if hema_point:
        vec_hema = median_od_vector_from_points(img_disp, [hema_point])
        vectors["hema"] = vec_hema
    if bg_point:
        vec_bg = median_od_vector_from_points(img_disp, [bg_point])
        vectors["bg"] = vec_bg

    if "aec" in vectors and "hema" in vectors:
        M = make_stain_matrix(vectors["aec"], vectors["hema"], vectors.get("bg"))
        deconv_img = deconvolve(img_disp, M)
        # AEC + HEMA Kanäle kombinieren
        combined = np.clip(deconv_img[:,:,0] + deconv_img[:,:,1], 0, None)
        thresh = (combined > np.percentile(combined, 90)).astype(np.uint8)*255
        all_points = get_centers(thresh, min_area=min_area)
    return all_points

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (OD/Deconv)", layout="wide")
st.title("🧬 Zellkern-Zähler – OD-Deconvolution, 1-Klick-Kalibrierung")

# -------------------- Session State --------------------
default_keys = ["aec_cal_points", "hema_cal_points", "bg_cal_points", "manual_points",
                "aec_vec", "hema_vec", "bg_vec", "last_file", "disp_width"]
for key in default_keys:
    if key not in st.session_state:
        st.session_state[key] = [] if "points" in key else None
st.session_state.disp_width = st.session_state.get("disp_width", 1400)

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.aec_cal_points = []
    st.session_state.hema_cal_points = []
    st.session_state.bg_cal_points = []
    st.session_state.manual_points = []
    st.session_state.last_file = uploaded_file.name

# -------------------- Bild vorbereiten --------------------
DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig*scale)), interpolation=cv2.INTER_AREA)

# -------------------- Sidebar: Modus --------------------
mode = st.sidebar.radio("Modus", ["Kalibrierpunkt setzen", "Manuell hinzufügen", "Punkt löschen"])
circle_radius = st.sidebar.slider("Kreisradius", 2, 15, 5)
min_area = st.sidebar.number_input("📏 Mindestfläche (px)", 5, 500, 10)

# -------------------- Klicklogik --------------------
coords = streamlit_image_coordinates(Image.fromarray(image_disp), key=f"clickable_image_{uploaded_file.name}", width=DISPLAY_WIDTH)
if coords:
    x, y = int(coords["x"]), int(coords["y"])
    if mode == "Kalibrierpunkt setzen":
        if not st.session_state.aec_cal_points:
            st.session_state.aec_cal_points.append((x,y))
            st.info(f"📍 AEC Kalibrierpunkt gesetzt ({x},{y})")
        elif not st.session_state.hema_cal_points:
            st.session_state.hema_cal_points.append((x,y))
            st.info(f"📍 HEMA Kalibrierpunkt gesetzt ({x},{y})")
        elif not st.session_state.bg_cal_points:
            st.session_state.bg_cal_points.append((x,y))
            st.info(f"📍 Hintergrundpunkt gesetzt ({x},{y})")
    elif mode == "Manuell hinzufügen":
        st.session_state.manual_points.append((x,y))
        st.info(f"✋ Manuell hinzugefügt ({x},{y})")
    elif mode == "Punkt löschen":
        for k in ["aec_cal_points","hema_cal_points","bg_cal_points","manual_points"]:
            st.session_state[k] = [p for p in st.session_state[k] if np.linalg.norm(np.array(p)-(x,y))>circle_radius]
        st.info("Punkt(e) gelöscht")

# -------------------- Kern-Erkennung --------------------
all_points = iterative_calib(
    image_disp,
    aec_point=st.session_state.aec_cal_points[0] if st.session_state.aec_cal_points else None,
    hema_point=st.session_state.hema_cal_points[0] if st.session_state.hema_cal_points else None,
    bg_point=st.session_state.bg_cal_points[0] if st.session_state.bg_cal_points else None,
    min_area=min_area
)

# alle Punkte zusammen
all_points += st.session_state.manual_points

# -------------------- Anzeige --------------------
marked_disp = image_disp.copy()
for (x,y) in all_points:
    cv2.circle(marked_disp, (x,y), circle_radius, (0,0,255), 2)

st.image(marked_disp, caption=f"Alle Kerne erkannt ({len(all_points)})", use_column_width=True)
st.metric("📊 Anzahl erkannter Kerne", len(all_points))

# -------------------- CSV Export --------------------
if all_points:
    df = pd.DataFrame(all_points, columns=["X_display","Y_display"])
    df["X_original"] = (df["X_display"]/scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"]/scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="zellkerne_od.csv", mime="text/csv")
