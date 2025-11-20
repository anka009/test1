import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Organoiden Zähler", layout="wide")
st.title("🧬 Organoiden Zähler")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # --- Sidebar-Parameter ---
    st.sidebar.header("⚙️ Parameter")
    clip_limit = st.sidebar.slider("CLAHE Kontrastverstärkung", 1.0, 5.0, 2.0, 0.1)
    threshold_val = st.sidebar.slider("Threshold (Otsu-Offset)", -50, 50, 0, 1)
    min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 10000, 1000, 10)

    # --- Vorverarbeitung ---
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, otsu + threshold_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Konturen finden ---
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_size:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

    # --- Markierungseinstellungen ---
    st.sidebar.header("🎨 Markierung")
    color = st.sidebar.color_picker("Farbe", "#FF0000")
    bgr_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
    radius = st.sidebar.slider("Radius", 1, 20, 5)
    line_thickness = st.sidebar.slider("Linienstärke", 1, 5, 2)

    # --- CSV-Export ---
    df = pd.DataFrame(centers, columns=["x", "y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", csv, "organoiden.csv", "text/csv")

    # --- Manuelle Zählung & Löschung über Bildklicks ---
    st.header("🖱️ Manuelle Bearbeitung")

    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), radius, bgr_color, line_thickness)

    coords = streamlit_image_coordinates(Image.fromarray(marked))

    if coords is not None:
        st.write(f"Klick erkannt bei: ({coords['x']}, {coords['y']})")
        action = st.radio("Aktion wählen:", ["Organid hinzufügen", "Organid löschen"])

        if action == "Organid hinzufügen":
            centers.append((coords["x"], coords["y"]))
            st.success("Organid manuell hinzugefügt ✅")

        elif action == "Organid löschen":
            def remove_nearest(center_list, click, max_dist=20):
                cx, cy = click["x"], click["y"]
                return [(x, y) for (x, y) in center_list
                        if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) > max_dist]
            centers = remove_nearest(centers, coords)
            st.success("Nächstgelegener Organid entfernt 🗑️")

    # --- aktualisierte Anzeige ---
    updated = image.copy()
    for (x, y) in centers:
        cv2.circle(updated, (x, y), radius, bgr_color, line_thickness)

    st.image(updated, caption=f"Aktuelle Organiden: {len(centers)}", use_container_width=True)
