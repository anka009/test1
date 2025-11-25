import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

# -------------------- Hilfsfunktionen --------------------

def is_near(p1, p2, r=5):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, existing_points, min_dist=5):
    out = []
    for p in points:
        if not any(is_near(p, e, min_dist) for e in existing_points):
            out.append(p)
    return out

def extract_patch(img, x, y, radius=5):
    y_min = max(0, y - radius)
    y_max = min(img.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(img.shape[1], x + radius + 1)
    return img[y_min:y_max, x_min:x_max]

def median_od_vector_from_patch(patch):
    patch = patch.astype(np.float32)
    od = -np.log((patch + 1) / 255)
    vec = np.median(od.reshape(-1,3), axis=0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

def normalize_vector(v):
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def make_stain_matrix(aec_vec, hema_vec, bg_vec=None):
    aec_vec = normalize_vector(aec_vec)
    hema_vec = normalize_vector(hema_vec)
    if bg_vec is None:
        bg_vec = np.cross(aec_vec, hema_vec)
        bg_vec = normalize_vector(bg_vec)
    else:
        bg_vec = normalize_vector(bg_vec)
    M = np.stack([aec_vec, hema_vec, bg_vec], axis=1)
    M += np.eye(3) * 1e-6
    return M

def deconvolve(img_rgb, stain_matrix):
    img_rgb = img_rgb.astype(np.float32)
    OD = -np.log((img_rgb + 1) / 255)
    M_inv = np.linalg.pinv(stain_matrix)
    C = OD.reshape(-1,3) @ M_inv.T
    return C.reshape(img_rgb.shape)

def detect_nuclei(deconv_channel, threshold=0.2, min_area=5):
    mask = (deconv_channel > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers

# -------------------- Session State --------------------
for key in ["groups", "all_points", "last_file", "disp_width"]:
    if key not in st.session_state:
        if key in ["groups", "all_points"]:
            st.session_state[key] = []
        elif key == "disp_width":
            st.session_state[key] = 800
        else:
            st.session_state[key] = None

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg","png","tif"])
if not uploaded_file:
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups = []
    st.session_state.all_points = []
    st.session_state.last_file = uploaded_file.name

DISPLAY_WIDTH = st.slider("Bildbreite", 200, 2000, st.session_state.disp_width)
st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H, W = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H*scale)), interpolation=cv2.INTER_AREA)

# -------------------- Klickmodus --------------------
st.sidebar.markdown("### Modus")
mode = st.sidebar.radio("Aktion", ["Kalibriere und zähle Kern", "Punkt löschen"])

calib_patch_radius = st.sidebar.slider("Kalibrier-Radius (px)", 1, 10, 5)
circle_radius = st.sidebar.slider("Kreisradius", 1, 10, 5)
threshold = st.sidebar.slider("OD-Threshold für Kern", 0.05, 1.0, 0.2, 0.01)
min_area = st.sidebar.slider("Min. Fläche (px)", 1, 50, 5)

# -------------------- Anzeige bisher gezählter Punkte --------------------
marked_disp = image_disp.copy()
for group in st.session_state.groups:
    for (x, y) in group["points"]:
        cv2.circle(marked_disp, (x, y), circle_radius, group["color"], -1)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key=f"clickable_image_{st.session_state.last_file}", width=DISPLAY_WIDTH)

# -------------------- Klicklogik --------------------
if coords:
    x, y = int(coords["x"]), int(coords["y"])
    if mode == "Punkt löschen":
        st.session_state.all_points = [p for p in st.session_state.all_points if not is_near(p, (x,y), circle_radius)]
        for g in st.session_state.groups:
            g["points"] = [p for p in g["points"] if not is_near(p,(x,y),circle_radius)]
        st.info("Punkt(e) gelöscht")
    else:
        patch = extract_patch(image_disp, x, y, calib_patch_radius)
        vec = median_od_vector_from_patch(patch)
        # Echte OD-Deconvolution AEC/Hema
        M = make_stain_matrix(vec, vec)  # AEC/Hema hier gleich für Demo
        deconv = deconvolve(image_disp, M)
        channel = deconv[:,:,0]  # erste Stain-Komponente
        detected = detect_nuclei(channel, threshold=threshold, min_area=min_area)
        new_points = dedup_points(detected, st.session_state.all_points, min_dist=circle_radius)
        if new_points:
            st.session_state.all_points.extend(new_points)
            st.session_state.groups.append({
                "vec": vec,
                "points": new_points,
                "color": tuple(np.random.randint(0,255,3).tolist())
            })
            st.success(f"Neue Kerne gezählt: {len(new_points)}")

# -------------------- Anzeige --------------------
marked_disp = image_disp.copy()
for group in st.session_state.groups:
    for (x, y) in group["points"]:
        cv2.circle(marked_disp, (x, y), circle_radius, group["color"], -1)

st.image(marked_disp, caption="Gezählte Kerne", use_column_width=True)

# -------------------- Zusammenfassung --------------------
st.markdown("### Zusammenfassung")
total_unique = len(st.session_state.all_points)
st.write(f"💠 Gesamtkerne (unique): {total_unique}")

for i, group in enumerate(st.session_state.groups):
    st.write(f"Gruppe {i+1} – Kerne: {len(group['points'])}")

# -------------------- CSV Export --------------------
if st.session_state.all_points:
    rows = []
    for i, group in enumerate(st.session_state.groups):
        for x, y in group["points"]:
            rows.append({"X_display": x, "Y_display": y, "Group": i+1})
    df = pd.DataFrame(rows)
    st.download_button("CSV exportieren", df.to_csv(index=False).encode("utf-8"), file_name="kerne.csv")
