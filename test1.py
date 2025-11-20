import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Organoiden Zähler", layout="wide")
st.title("🧬 Organoiden Zähler")

uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # --- Automatische Erkennung ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, otsu_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    auto_centers = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:  # Mindestfläche
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                auto_centers.append((cx, cy))

    # --- Session-State initialisieren ---
    if "centers" not in st.session_state:
        st.session_state.centers = auto_centers.copy()

    # --- Button für manuelle Korrektur ---
    if st.button("Manuelle Korrektur starten"):
        st.session_state.edit_mode = True

    if st.session_state.get("edit_mode", False):
        st.header("🖱️ Manuelle Bearbeitung")

        # Bild mit aktuellen Zentren
        marked = image.copy()
        for (x, y) in st.session_state.centers:
            cv2.circle(marked, (x, y), 8, (0, 0, 255), 2)

        coords = streamlit_image_coordinates(Image.fromarray(marked))
        if coords is not None:
            x0, y0 = coords["x"], coords["y"]
            action = st.radio("Aktion wählen:", ["Organid hinzufügen", "Organid löschen"], horizontal=True)

            if action == "Organid hinzufügen":
                st.session_state.centers.append((x0, y0))
                st.success("Organid hinzugefügt ✅")

            elif action == "Organid löschen":
                st.session_state.centers = [(x, y) for (x, y) in st.session_state.centers
                                            if np.hypot(x - x0, y - y0) > 20]
                st.success("Organid entfernt 🗑️")

    # --- Finale Anzeige ---
    updated = image.copy()
    for (x, y) in st.session_state.centers:
        cv2.circle(updated, (x, y), 8, (0, 0, 255), 2)

    st.image(updated, caption=f"Aktuelle Organiden: {len(st.session_state.centers)}", use_container_width=True)

    # --- Export ---
    df = pd.DataFrame(st.session_state.centers, columns=["X", "Y"])
    st.download_button("📥 CSV exportieren", df.to_csv(index=False).encode("utf-8"),
                       file_name="organoiden.csv", mime="text/csv")
