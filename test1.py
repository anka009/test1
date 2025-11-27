# canvas_iterative_deconv_v3.py
"""
V.3 — Robustere, schnellere Iterative Kern-Zählung
Changelog (gegenüber v2):
- stabilere Stain-Matrix via orthogonalisierung (SVD/Gram-Schmidt fallback)
- robustere Patch-OD-Schätzung (Top-perzentil + median)
- Sauvola-Threshold (falls skimage verfügbar) mit Fallback auf adaptives CV2
- Mask-Berechnung auf downsampled Kanal + Rückskalierung der Zentren (viel schneller)
- Dedup via KDTree (scipy) mit Numpy-Fallback (sort+window) — vermeidet O(n^2)
- Undo/History speichert präzise Gruppen-Zuordnung beim Löschen
- Precomputed log LUT für schnellere Deconvolution
- mehr defensive checks + klarere Status-Messages

Benutzt: streamlit, numpy, cv2, PIL, pandas. Optional: skimage, scipy
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from pathlib import Path
import math
import json

# optional imports
try:
    from skimage.filters import threshold_sauvola
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv) — v3", layout="wide")
st.title("🧬 Iterative Kern-Zählung — V.3 (robust & faster)")

# -------------------- Utilities --------------------
def is_near(p1, p2, r=6.0):
    return np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)) < float(r)


def dedup_new_points_kdtree(candidates, existing, min_dist=6.0):
    """Return list of candidates that are at least min_dist away from any existing point.
    Uses cKDTree when available; otherwise falls back to a fast numpy windowed method.
    Points are (x,y) tuples.
    """
    if not candidates:
        return []
    cand = np.array(candidates, dtype=float)
    if existing and SCIPY_AVAILABLE:
        tree = cKDTree(np.array(existing, dtype=float))
        dists, _ = tree.query(cand, k=1, n_jobs=-1)
        mask = dists >= min_dist
        return [tuple(map(int, map(round, p))) for p in cand[mask]]
    elif not existing:
        return [tuple(map(int, map(round, p))) for p in cand]
    else:
        # fallback: sort by x and check window (O(n log n) + local checks)
        ex = np.array(existing, dtype=float)
        ex_sorted_idx = np.argsort(ex[:, 0])
        ex_sorted = ex[ex_sorted_idx]
        out = []
        # helper to check if candidate is near any existing via small window
        for p in cand:
            x = p[0]
            # binary search window in ex_sorted where x difference < min_dist
            lo = np.searchsorted(ex_sorted[:, 0], x - min_dist, side='left')
            hi = np.searchsorted(ex_sorted[:, 0], x + min_dist, side='right')
            neigh = ex_sorted[lo:hi]
            if neigh.size == 0:
                out.append((int(round(p[0])), int(round(p[1]))))
            else:
                if np.all(np.linalg.norm(neigh - p, axis=1) >= min_dist):
                    out.append((int(round(p[0])), int(round(p[1]))))
        return out


def extract_patch(img, x, y, radius=5):
    y_min = max(0, int(y - radius))
    y_max = min(img.shape[0], int(y + radius + 1))
    x_min = max(0, int(x - radius))
    x_max = min(img.shape[1], int(x + radius + 1))
    if y_min >= y_max or x_min >= x_max:
        return None
    return img[y_min:y_max, x_min:x_max]


def median_od_vector_from_patch(patch, top_pct=0.35, eps=1e-6):
    """Compute a robust OD vector from a small RGB patch.
    Strategy: compute OD per pixel, compute OD-norm per pixel, take top percentile
    (strongest absorbing pixels) and compute median across them.
    Returns normalized 3-vector (float32) or None on failure.
    """
    if patch is None or patch.size == 0:
        return None
    patch = patch.astype(np.float32)
    # compute OD
    with np.errstate(divide='ignore', invalid='ignore'):
        OD = -np.log(np.clip((patch + eps) / 255.0, 1e-8, 1.0))
    norms = np.linalg.norm(OD.reshape(-1, 3), axis=1)
    if np.all(np.isfinite(norms) == False) or np.nanmax(norms) <= 1e-8:
        return None
    # pick top_pct of pixels by OD norm
    k = max(1, int(len(norms) * float(top_pct)))
    idx = np.argpartition(-norms, k - 1)[:k]
    vec = np.median(OD.reshape(-1, 3)[idx, :], axis=0)
    if not np.all(np.isfinite(vec)):
        return None
    n = np.linalg.norm(vec)
    if n <= 1e-8:
        return None
    return (vec / n).astype(np.float32)


def normalize_vector(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 1e-12:
        return v.astype(float)
    return (v / n).astype(float)


def make_stain_matrix(target_vec, hema_vec, bg_vec=None):
    """Create numerically stable 3x3 stain matrix.
    - target_vec, hema_vec: 3-element iterables
    - if bg_vec None, compute orthogonal via SVD/Gram-Schmidt
    Returns 3x3 float32 matrix with small regularization.
    """
    t = normalize_vector(target_vec)
    h = normalize_vector(hema_vec)

    # if t and h almost collinear, perturb h slightly
    if abs(np.dot(t, h)) > 0.995:
        # small orthogonal perturbation
        h = normalize_vector(h + 1e-2 * np.array([0.1, -0.07, 0.05]))

    if bg_vec is None:
        # stack and perform SVD to get orthogonal basis
        try:
            A = np.column_stack([t, h])
            U, S, Vt = np.linalg.svd(A, full_matrices=True)
            # pick third column of U as orthogonal complement
            if U.shape[1] >= 3:
                bg = U[:, 2]
            else:
                # fallback to Gram-Schmidt
                proj = np.dot(t, h) * t
                orth = h - proj
                if np.linalg.norm(orth) < 1e-6:
                    # final fallback: cross-product
                    bg = np.cross(t, h)
                else:
                    bg = orth
        except Exception:
            bg = np.cross(t, h)
            if np.linalg.norm(bg) < 1e-6:
                # tiny perturbation fallback
                if abs(t[0]) > 0.1 or abs(t[1]) > 0.1:
                    bg = np.array([t[1], -t[0], 0.0], dtype=float)
                else:
                    bg = np.array([0.0, t[2], -t[1]], dtype=float)
    else:
        bg = normalize_vector(bg_vec)

    bg = normalize_vector(bg)
    M = np.column_stack([t, h, bg]).astype(np.float32)
    M = M + np.eye(3, dtype=np.float32) * 1e-8
    return M


# Precompute log LUT for speed (0..255)
_LOG_LUT = -np.log((np.arange(256).astype(np.float32) + 1e-6) / 255.0)


def deconvolve(img_rgb, M):
    """Fast deconvolution using precomputed LUT. Returns C of shape (H,W,3) float32.
    Expects img_rgb uint8 RGB.
    """
    if img_rgb.dtype != np.uint8:
        img = img_rgb.astype(np.uint8)
    else:
        img = img_rgb
    H, W = img.shape[:2]
    # use LUT per channel then stack
    img_r = _LOG_LUT[img[:, :, 0]]
    img_g = _LOG_LUT[img[:, :, 1]]
    img_b = _LOG_LUT[img[:, :, 2]]
    OD = np.stack([img_r, img_g, img_b], axis=2).reshape(-1, 3)  # N x 3
    try:
        pinv = np.linalg.pinv(M)
        C = (pinv @ OD.T).T
    except Exception:
        return None
    return C.reshape(H, W, 3).astype(np.float32)


def detect_centers_from_channel(channel, min_area=8, downsample=4, sauvola_window=35, sauvola_k=0.2):
    """Robuste Erkennung basierend auf einem real-valued channel (float);
    - downsample: compute mask at reduced resolution to speed up processing
    - returns list of centers in original coordinates and mask (resized to original)
    """
    arr = np.array(channel, dtype=np.float32)
    arr = np.maximum(arr, 0.0)
    H, W = arr.shape
    if H < 1 or W < 1:
        return [], np.zeros_like(arr, dtype=np.uint8)

    # robust normalization
    vmin, vmax = np.percentile(arr, [2, 99.5])
    if vmax - vmin < 1e-6:
        return [], np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

    # downsample for mask creation
    ds = max(1, int(downsample))
    small = cv2.resize((norm * 255.0).astype(np.uint8), (W // ds, H // ds), interpolation=cv2.INTER_AREA)

    # thresholding: prefer Sauvola if available
    if SKIMAGE_AVAILABLE and min(small.shape) >= 16:
        try:
            small_float = small.astype(np.float32) / 255.0
            thresh = threshold_sauvola(small_float, window_size=min(sauvola_window, min(small.shape) - 1), k=sauvola_k)
            mask_small = (small_float > thresh).astype(np.uint8) * 255
        except Exception:
            _, mask_small = cv2.threshold(small, int(0.5 * 255), 255, cv2.THRESH_BINARY)
    else:
        # fallback: adaptiveThreshold with blockSize proportional to small image
        bsize = max(3, (min(small.shape) // 20) | 1)  # odd
        C = 2
        try:
            mask_small = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, bsize, C)
        except Exception:
            _, mask_small = cv2.threshold(small, int(0.5 * 255), 255, cv2.THRESH_BINARY)

    # small morphological cleanups on small mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel)

    # upscale to original size for contour detection
    mask_up = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # find contours at full resolution
    contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= max(1, min_area):
            M = cv2.moments(c)
            if M.get('m00', 0) != 0:
                cx = float(M['m10'] / M['m00'])
                cy = float(M['m01'] / M['m00'])
                centers.append((int(round(cx)), int(round(cy))))
    return centers, mask_up


# -------------------- SessionState initialisierung --------------------
for k in [
    "groups",
    "all_points",
    "last_file",
    "disp_width",
    "C_cache",
    "last_M_hash",
    "history",
    "params_hash",
]:
    if k not in st.session_state:
        if k in ["groups", "all_points", "history"]:
            st.session_state[k] = []
        elif k == "disp_width":
            st.session_state[k] = 1000
        else:
            st.session_state[k] = None

# -------------------- UI: Upload + Parameter --------------------
uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# reset on new file
if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups = []
    st.session_state.all_points = []
    st.session_state.C_cache = None
    st.session_state.last_M_hash = None
    st.session_state.history = []
    st.session_state.last_file = uploaded_file.name

col1, col2 = st.columns([2, 1])
with col2:
    st.sidebar.markdown("### Parameter")
    calib_radius = st.sidebar.slider("Kalibrier-Radius (px, original image)", 1, 60, 6)
    detection_threshold = st.sidebar.slider("Threshold (nur Info, 0-1)", 0.01, 0.9, 0.2, 0.01)
    min_area_display = st.sidebar.number_input("Min. Konturfläche (px) — angezeigt (Display-Äquivalent)", min_value=1, max_value=2000, value=80)
    dedup_dist_display = st.sidebar.slider("Min. Distanz für Doppelzählung (px, Display)", 1, 80, 12)
    circle_radius = st.sidebar.slider("Marker-Radius (px, Display)", 1, 14, 6)
    downsample_mask = st.sidebar.select_slider("Mask Downsample-Faktor", options=[1,2,4,8], value=4)
    st.sidebar.markdown("### Startvektoren (optional, RGB)")
    hema_default = st.sidebar.text_input("Hematoxylin vector (comma)", value="0.65,0.70,0.29")
    aec_default = st.sidebar.text_input("Chromogen (e.g. AEC/DAB) vector (comma)", value="0.27,0.57,0.78")

    # parse start vectors
    try:
        hema_vec0 = np.array([float(x.strip()) for x in hema_default.split(",")], dtype=float)
        aec_vec0 = np.array([float(x.strip()) for x in aec_default.split(",")], dtype=float)
    except Exception:
        hema_vec0 = np.array([0.65, 0.70, 0.29], dtype=float)
        aec_vec0 = np.array([0.27, 0.57, 0.78], dtype=float)

with col1:
    DISPLAY_WIDTH = st.slider("Anzeige-Breite (px)", 300, 1600, st.session_state.disp_width)
    st.session_state.disp_width = DISPLAY_WIDTH

# -------------------- Prepare images --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / float(W_orig)
H_disp = int(round(H_orig * scale))
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, H_disp), interpolation=cv2.INTER_AREA)

# convert units
area_scale = (1.0 / (scale * scale)) if scale > 0 else 1.0
min_area_orig = max(1, int(round(min_area_display * area_scale)))
dedup_dist_orig = max(1.0, float(dedup_dist_display / scale))

# draw existing points
display_canvas = image_disp.copy()
PRESET_COLORS = [
    (220, 20, 60),    # crimson
    (0, 128, 0),      # green
    (30, 144, 255),   # dodger
    (255, 165, 0),    # orange
    (148, 0, 211),    # purple
    (0, 255, 255),    # cyan
]
for i, g in enumerate(st.session_state.groups):
    col = tuple(int(x) for x in g.get("color", PRESET_COLORS[i % len(PRESET_COLORS)]))
    for (x_orig, y_orig) in g["points"]:
        x_disp = int(round(x_orig * scale))
        y_disp = int(round(y_orig * scale))
        cv2.circle(display_canvas, (x_disp, y_disp), circle_radius, col, -1)
    if g["points"]:
        px_disp = int(round(g["points"][0][0] * scale))
        py_disp = int(round(g["points"][0][1] * scale))
        cv2.putText(display_canvas, f"G{i+1}:{len(g['points'])}", (px_disp + 6, py_disp - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

from streamlit_image_coordinates import streamlit_image_coordinates
coords = streamlit_image_coordinates(Image.fromarray(display_canvas),
                                    key=f"clickable_image_v3_{st.session_state.last_file}",
                                    width=DISPLAY_WIDTH)

# Sidebar actions
mode = st.sidebar.radio("Aktion", ["Kalibriere und zähle Gruppe (Klick)", "Punkt löschen", "Undo letzte Aktion"]) 
st.sidebar.markdown("---")
if st.sidebar.button("Reset (Alle Gruppen)"):
    st.session_state.history.append(("reset", {
        "groups": json.loads(json.dumps(st.session_state.groups)),
        "all_points": list(st.session_state.all_points)
    }))
    st.session_state.groups = []
    st.session_state.all_points = []
    st.session_state.C_cache = None
    st.session_state.last_M_hash = None
    st.success("Zurückgesetzt.")

# -------------------- Click handling --------------------
if coords:
    x_disp, y_disp = int(coords["x"]), int(coords["y"])
    x_orig = int(round(x_disp / scale))
    y_orig = int(round(y_disp / scale))

    if mode == "Punkt löschen":
        removed = []
        new_all = []
        # build groups mapping for history
        removed_map = []  # list of (group_idx, point)
        for gi, g in enumerate(st.session_state.groups):
            kept = []
            for p in g["points"]:
                if is_near(p, (x_orig, y_orig), dedup_dist_orig):
                    removed.append(p)
                    removed_map.append({"group_idx": gi, "point": p})
                else:
                    kept.append(p)
            g["points"] = kept
        for p in st.session_state.all_points:
            if not is_near(p, (x_orig, y_orig), dedup_dist_orig):
                new_all.append(p)
        if removed:
            st.session_state.history.append(("delete_points", {"removed_map": removed_map}))
            st.session_state.all_points = new_all
            st.success(f"{len(removed)} Punkt(e) gelöscht.")
        else:
            st.info("Kein Punkt in der Nähe gefunden.")

    elif mode == "Undo letzte Aktion":
        if st.session_state.history:
            action, payload = st.session_state.history.pop()
            if action == "add_group":
                idx = payload["group_idx"]
                if 0 <= idx < len(st.session_state.groups):
                    grp = st.session_state.groups.pop(idx)
                    for pt in grp["points"]:
                        # remove one instance from all_points
                        if pt in st.session_state.all_points:
                            st.session_state.all_points.remove(pt)
                    st.success("Letzte Gruppen-Aktion rückgängig gemacht.")
                else:
                    st.warning("Letzte Aktion konnte nicht rückgängig gemacht werden.")
            elif action == "delete_points":
                removed_map = payload.get("removed_map", [])
                # restore points into their original groups
                # ensure groups exist (they should), otherwise append
                for item in removed_map:
                    gi = item["group_idx"]
                    pt = tuple(item["point"]) if isinstance(item["point"], list) else item["point"]
                    if 0 <= gi < len(st.session_state.groups):
                        st.session_state.groups[gi]["points"].append(pt)
                    else:
                        st.session_state.groups.append({"vec": None, "points": [pt], "color": PRESET_COLORS[len(st.session_state.groups) % len(PRESET_COLORS)]})
                    st.session_state.all_points.append(pt)
                st.success("Gelöschte Punkte wiederhergestellt.")
            elif action == "reset":
                st.session_state.groups = payload["groups"]
                st.session_state.all_points = payload["all_points"]
                st.success("Reset rückgängig gemacht.")
            else:
                st.warning("Undo: unbekannte Aktion.")
        else:
            st.info("Keine Aktion zum Rückgängig machen.")

    else:
        # Mode: Kalibriere und zähle Gruppe
        patch = extract_patch(image_orig, x_orig, y_orig, calib_radius)
        vec = median_od_vector_from_patch(patch)
        if vec is None:
            st.warning("Patch unbrauchbar (zu homogen oder außerhalb). Bitte anders klicken.")
        else:
            M = make_stain_matrix(vec, hema_vec0)
            M_hash = tuple(np.round(M.flatten(), 6).tolist())

            recompute = False
            # also track mask downsample param & image name to decide cache
            params_hash = (st.session_state.last_file, M_hash, downsample_mask)
            if st.session_state.C_cache is None or st.session_state.last_M_hash != M_hash or st.session_state.params_hash != params_hash:
                recompute = True
            if recompute:
                C_full = deconvolve(image_orig, M)
                if C_full is None:
                    st.error("Deconvolution fehlgeschlagen (numerisch).")
                    st.stop()
                st.session_state.C_cache = C_full
                st.session_state.last_M_hash = M_hash
                st.session_state.params_hash = params_hash
            else:
                C_full = st.session_state.C_cache

            channel_full = C_full[:, :, 0]

            centers_orig, mask = detect_centers_from_channel(channel_full, min_area=min_area_orig, downsample=downsample_mask)

            new_centers = dedup_new_points_kdtree(centers_orig, st.session_state.all_points, min_dist=dedup_dist_orig)

            if new_centers:
                color = PRESET_COLORS[len(st.session_state.groups) % len(PRESET_COLORS)]
                group = {"vec": vec.tolist(), "points": new_centers, "color": color}
                st.session_state.history.append(("add_group", {"group_idx": len(st.session_state.groups)}))
                st.session_state.groups.append(group)
                st.session_state.all_points.extend(new_centers)
                st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_centers)}")
            else:
                st.info("Keine neuen Kerne (alle bereits gezählt oder keine Detektion).")

# -------------------- Ergebnisse & Export --------------------
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

    if st.button("Speichere Maske (letzte Deconv-Channel)"):
        if st.session_state.C_cache is not None:
            channel_to_save = st.session_state.C_cache[:, :, 0]
            vmin, vmax = np.percentile(channel_to_save, [2, 99.5])
            norm = np.clip((channel_to_save - vmin) / max(1e-8, (vmax - vmin)), 0.0, 1.0)
            u8 = (norm * 255).astype(np.uint8)
            u8_disp = cv2.resize(u8, (DISPLAY_WIDTH, H_disp), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(u8_disp)
            buf = st.session_state.last_file + "_channel_v3.png"
            pil.save(buf)
            with open(buf, "rb") as f:
                st.download_button("📥 Download Channel (PNG)", f.read(), file_name=buf, mime="image/png")
        else:
            st.info("Keine Deconvolution im Cache verfügbar.")

# CSV Export
if st.session_state.all_points:
    rows = []
    for i, g in enumerate(st.session_state.groups):
        for (x_orig, y_orig) in g["points"]:
            x_disp = int(round(x_orig * scale))
            y_disp = int(round(y_orig * scale))
            rows.append({"Group": i + 1, "X_display": int(x_disp), "Y_display": int(y_disp),
                         "X_original": int(x_orig), "Y_original": int(y_orig)})
    df = pd.DataFrame(rows)
    st.download_button("📥 CSV exportieren (Gruppen, inkl. Original-Koords)", df.to_csv(index=False).encode("utf-8"),
                       file_name="kern_gruppen_v3.csv", mime="text/csv")

    df_unique = pd.DataFrame(st.session_state.all_points, columns=["X_original", "Y_original"])
    df_unique["X_display"] = (df_unique["X_original"] * scale).round().astype(int)
    df_unique["Y_display"] = (df_unique["Y_original"] * scale).round().astype(int)
    st.download_button("📥 CSV exportieren (unique Gesamt)", df_unique.to_csv(index=False).encode("utf-8"),
                       file_name="kern_unique_v3.csv", mime="text/csv")

st.markdown("---")
st.caption("Hinweise: Deconvolution wird auf dem ORIGINALbild ausgeführt. V3 verwendet eine robuste OD-Schätzung, mask-downsampling und KDTree-basierte Dedup. Wenn skimage/scipy fehlen, laufen Fallback-Algorithmen.")
