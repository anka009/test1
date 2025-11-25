# canvas_iterative_deconv.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv)", layout="wide")
st.title("🧬 Iterative Kern-Zählung — Klick → Kalibriere → Zähle (ein Kern nur einmal)")

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=6):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_new_points(candidates, existing, min_dist=6):
    """Return candidates that are not within min_dist of any existing point."""
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
    patch = img[y_min:y_max, x_min:x_max]
    return patch

# -------------------- Robustere OD-Vektor-Bestimmung --------------------
def median_od_vector_from_patch(patch, top_fraction=0.25, eps=1e-6):
    """Compute normalized median OD vector using only strongest OD pixels."""
    if patch is None or patch.size == 0:
        return None
    patch = patch.astype(np.float32)
    OD = -np.log((patch + eps) / 255.0)  # OD
    flat = OD.reshape(-1, 3)
    mag = np.linalg.norm(flat, axis=1)
    k = max(5, int(len(mag) * top_fraction))
    idx = np.argsort(-mag)[:k]
    vec = np.median(flat[idx], axis=0)
    norm = np.linalg.norm(vec)
    if norm <= 1e-8 or np.any(np.isnan(vec)):
        return None
    return (vec / norm).astype(np.float32)

# -------------------- Kontur-basierte Kern-Detektion --------------------
def detect_centers_from_channel(channel, threshold=0.2, min_area=8, min_circularity=0.4):
    """
    Detect nuclei centers using contours, thresholding, morphology, and circularity filter.
    """
    arr = np.maximum(channel, 0.0)
    
    # robust percentile clipping
    vmin, vmax = np.percentile(arr, [1, 98])
    if vmax - vmin > 1e-6:
        norm = (arr - vmin) / (vmax - vmin)
    else:
        norm = arr - vmin
    mask = (norm >= threshold).astype(np.uint8) * 255
    
    # Morphology: Open → Close → Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < max(1, min_area):
            continue
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-9)
        if circularity < min_circularity:
            continue
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(round(M["m10"] / M["m00"]))
            cy = int(round(M["m01"] / M["m00"]))
            centers.append((cx, cy))
    return centers, mask

# -------------------- Session state --------------------
for k in ["groups", "all_points", "last_file", "disp_width"]:
    if k not in st.session_state:
        if k in ["groups", "all_points"]:
            st.session_state[k] = []
        elif k == "disp_width":
            st.session_state[k] = 1000
        else:
            st.session_state[k] = None

# -------------------- UI: Upload + params --------------------
uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups = []
    st.session_state.all_points = []
    st.session_state.last_file = uploaded_file.name

col1, col2 = st.columns([2, 1])
with col2:
    st.sidebar.markdown("### Parameter")
    calib_radius = st.sidebar.slider("Kalibrier-Radius (px)", 1, 10, 5)
    detection_threshold = st.sidebar.slider("Threshold (0-1) für Detektion (Percentil-normalisiert)", 0.01, 0.9, 0.2, 0.01)
    min_area = st.sidebar.number_input("Min. Konturfläche (px)", min_value=1, max_value=1000, value=8)
    dedup_dist = st.sidebar.slider("Min. Distanz für Doppelzählung (px)", 1, 20, 6)
    circle_radius = st.sidebar.slider("Marker-Radius (px)", 1, 10, 5)
    # default Hematoxylin & AEC vectors (Ruifrok-like approximations)
    st.sidebar.markdown("### Startvektoren (optional)")
    hema_default = st.sidebar.text_input("Hematoxylin vector (comma)", value="0.65,0.70,0.29")
    aec_default = st.sidebar.text_input("Chromogen (e.g. AEC/DAB) vector (comma)", value="0.27,0.57,0.78")
    # convert strings to arrays
    try:
        hema_vec0 = np.array([float(x.strip()) for x in hema_default.split(",")], dtype=float)
        aec_vec0 = np.array([float(x.strip()) for x in aec_default.split(",")], dtype=float)
    except Exception:
        hema_vec0 = np.array([0.65, 0.70, 0.29], dtype=float)
        aec_vec0 = np.array([0.27, 0.57, 0.78], dtype=float)

with col1:
    DISPLAY_WIDTH = st.slider("Anzeige-Breite (px)", 300, 1400, st.session_state.disp_width)
    st.session_state.disp_width = DISPLAY_WIDTH

# -------------------- Prepare images --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)

# -------------------- Draw existing points --------------------
display_canvas = image_disp.copy()
# draw groups with colors and labels
for i, g in enumerate(st.session_state.groups):
    col = tuple(int(x) for x in g["color"])
    for (x, y) in g["points"]:
        cv2.circle(display_canvas, (x, y), circle_radius, col, -1)
    # optionally draw label near first point
    if g["points"]:
        px, py = g["points"][0]
        cv2.putText(display_canvas, f"G{i+1}:{len(g['points'])}", (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

# image widget (unique key per file)
coords = streamlit_image_coordinates(Image.fromarray(display_canvas),
                                    key=f"clickable_image_{st.session_state.last_file}",
                                    width=DISPLAY_WIDTH)

# -------------------- Click logic --------------------
mode = st.sidebar.radio("Aktion", ["Kalibriere und zähle Gruppe (Klick)", "Punkt löschen"])

if coords:
    x, y = int(coords["x"]), int(coords["y"])

    if mode == "Punkt löschen":
        # remove from all_points and groups
        st.session_state.all_points = [p for p in st.session_state.all_points if not is_near(p, (x, y), dedup_dist)]
        for g in st.session_state.groups:
            g["points"] = [p for p in g["points"] if not is_near(p, (x, y), dedup_dist)]
        st.success("Punkt(e) gelöscht")

    else:  # Kalibriere und zähle Gruppe
        patch = extract_patch(image_disp, x, y, calib_radius)
        patch_brightness = patch.mean()
        if patch_brightness > 230:
            st.warning("Patch zu hell — bitte dunkleres Kernzentrum anklicken.")
        else:
            # 1) Robust OD-Vektor aus Patch
            vec = median_od_vector_from_patch(patch)
            if vec is None:
                st.warning("Patch unbrauchbar (zu homogen oder außerhalb). Bitte anders klicken.")
            else:
                # 2) Build stain matrix mit Hematoxylin default
                M = make_stain_matrix(vec, hema_vec0)
                # 3) Deconvolution
                C = deconvolve(image_disp, M)
                if C is None:
                    st.error("Deconvolution fehlgeschlagen (numerisch).")
                else:
                    # 4) Wähle Kanal mit stärkster Streuung (Target vs Hema)
                    c0_std = np.std(C[:, :, 0])
                    c1_std = np.std(C[:, :, 1])
                    use_ch = 0 if c0_std > c1_std else 1
                    channel = C[:, :, use_ch]

                    # 5) Konturbasierte Kern-Detektion
                    centers, mask = detect_centers_from_channel(channel, threshold=detection_threshold, min_area=min_area)

                    # 6) Neue Zentren deduplizieren (globale Punkte)
                    new_centers = dedup_new_points(centers, st.session_state.all_points, min_dist=dedup_dist)

                    if new_centers:
                        # 7) Gruppe hinzufügen mit zufälliger Farbe
                        color = tuple(int(v) for v in np.random.randint(60, 230, 3))
                        st.session_state.groups.append({
                            "vec": vec.tolist(),
                            "points": new_centers,
                            "color": color
                        })
                        # 8) Globale Punkte aktualisieren
                        st.session_state.all_points.extend(new_centers)
                        st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_centers)}")
                    else:
                        st.info("Keine neuen Kerne (alle bereits gezählt).")

# -------------------- Show results & summary --------------------
st.markdown("## Ergebnisse")
colA, colB = st.columns([2, 1])
with colA:
    st.image(display_canvas, caption="Gezählte Kerne (Gruppenfarben)", use_column_width=True)

with colB:
    st.markdown("### Zusammenfassung")
    st.write(f"🔹 Gruppen gesamt: {len(st.session_state.groups)}")
    for i, g in enumerate(st.session_state.groups):
        st.write(f"• Gruppe {i+1}: {len(g['points'])} neue Kerne")
    st.markdown(f"**Gesamt (unique Kerne): {len(st.session_state.all_points)}**")
    if st.button("Zurücksetzen (Alle Gruppen löschen)"):
        st.session_state.groups = []
        st.session_state.all_points = []
        st.success("Zurückgesetzt.")

# -------------------- CSV Export --------------------
if st.session_state.all_points:
    rows = []
    for i, g in enumerate(st.session_state.groups):
        for (x, y) in g["points"]:
            rows.append({"Group": i + 1, "X_display": int(x), "Y_display": int(y)})
    df = pd.DataFrame(rows)
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren (Gruppen)", df.to_csv(index=False).encode("utf-8"),
                       file_name="kern_gruppen.csv", mime="text/csv")

    # also provide a summary CSV with unique points (no group)
    df_unique = pd.DataFrame(st.session_state.all_points, columns=["X_display", "Y_display"])
    df_unique["X_original"] = (df_unique["X_display"] / scale).round().astype("Int64")
    df_unique["Y_original"] = (df_unique["Y_display"] / scale).round().astype("Int64")
    st.download_button("📥 CSV exportieren (unique Gesamt)", df_unique.to_csv(index=False).encode("utf-8"),
                       file_name="kern_unique.csv", mime="text/csv")
