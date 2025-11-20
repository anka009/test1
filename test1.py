import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

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

    # --- CLAHE für besseren Kontrast ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Threshold mit Otsu ---
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(gray, otsu_thresh + threshold_val, 255, cv2.THRESH_BINARY)

    # --- Invertieren falls Kerne dunkel ---
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # --- Morphologische Filter ---
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Konturen finden ---
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Filtern nach Größe ---
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]

    # --- Mittelpunktberechnung ---
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    
    # --- Bild markieren ---
    # 🛠 Benutzeroptionen für die Markierung
    st.sidebar.header("🔧 Markierungseinstellungen")
    radius = st.sidebar.slider("Kreisradius", 2, 100, 8)
    line_thickness = st.sidebar.slider("Liniendicke", 1, 30, 2)
    color = st.sidebar.color_picker("Farbe der Markierung", "#ff0000")  # Standard: Rot

    # 🎨 Farbe konvertieren von Hex zu BGR
    hex_color = color.lstrip("#")
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Hex → RGB
    bgr_color = rgb_color[::-1]  # RGB → BGR für OpenCV


    

    # --- CSV Export ---
    df = pd.DataFrame(centers, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

    
    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), radius, bgr_color, line_thickness)

    st.image(marked, caption=f"Gefundene Zellkerne: {len(centers)}", use_container_width=True)

# --- Manuelle Zählung & Löschung über Bildklicks ---
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.header("🖱️ Manuelle Bearbeitung")

# Bild mit automatisch erkannten Organiden vorbereiten
marked = image.copy()
for (x, y) in centers:
    cv2.circle(marked, (x, y), radius, bgr_color, line_thickness)

# Übergabe an streamlit_image_coordinates (PIL-Image!)
coords = streamlit_image_coordinates(Image.fromarray(marked))

if coords is not None:
    st.write(f"Klick erkannt bei: ({coords['x']}, {coords['y']})")

    # Auswahl: Hinzufügen oder Löschen
    action = st.radio("Aktion wählen:", ["Organid hinzufügen", "Organid löschen"])

    if action == "Organid hinzufügen":
        centers.append((coords["x"], coords["y"]))
        st.success("Organid manuell hinzugefügt ✅")

    elif action == "Organid löschen":
        # Nächstgelegenen Punkt zu Klick finden und entfernen
        def remove_nearest(center_list, click, max_dist=20):
            cx, cy = click["x"], click["y"]
            new_list = []
            for (x, y) in center_list:
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist > max_dist:  # nur behalten, wenn nicht nah am Klick
                    new_list.append((x, y))
            return new_list

        centers = remove_nearest(centers, coords)
        st.success("Nächstgelegener Organid entfernt 🗑️")

# Aktualisiertes Bild mit neuen Zentren anzeigen
updated = image.copy()
for (x, y) in centers:
    cv2.circle(updated, (x, y), radius, bgr_color, line_thickness)

st.image(updated, caption=f"Aktuelle Organiden: {len(centers)}", use_container_width=True)
