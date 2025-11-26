# canvas_iterative_deconv_blobs_v2.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from skimage.feature import blob_log
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Iterative Kern-Zählung (OD + Deconv + Blobs)", layout="wide")
st.title("🧬 Iterative Kern-Zählung — Blob-Version (skimage.feature.blob_log)")

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=6):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_new_points(candidates, existing, min_dist=6):
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
    return img[y_min:y_max, x_min:x_max]

def median_od_vector_from_patch(patch, eps=1e-6):
    if patch is None or patch.size == 0:
        return None
    patch = patch.astype(np.float32)
    OD = -np.log(np.clip((patch + eps) / 255.0, 1e-8, 1.0))
    vec = np.median(OD.reshape(-1,3), axis=0)
    norm = np.linalg.norm(vec)
    if norm < 1e-8 or np.any(np.isnan(vec)):
        return None
    return (vec / norm).astype(np.float32)

def normalize_vector(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return (v / n).astype(float) if n > 1e-12 else v

def make_stain_matrix(target_vec, hema_vec, bg_vec=None):
    t = normalize_vector(target_vec)
    h = normalize_vector(hema_vec)
    if bg_vec is None:
        bg = np.cross(t,h)
        if np.linalg.norm(bg) < 1e-6:
            bg = np.array([t[1], -t[0], 0.0])
        bg = normalize_vector(bg)
    else:
        bg = normalize_vector(bg_vec)
    M = np.column_stack([t,h,bg]).astype(np.float32)
    M += np.eye(3) * 1e-8
    return M

def deconvolve(img_rgb, M):
    img = img_rgb.astype(np.float32)
    OD = -np.log(np.clip((img + 1e-6)/255.0, 1e-8, 1.0)).reshape(-1,3)
    pinv = np.linalg.pinv(M)
    C = (pinv @ OD.T).T
    return C.reshape(img_rgb.shape)

def detect_blobs(channel, min_sigma=2, max_sigma=6, num_sigma=5, threshold=0.03):
    arr = np.array(channel, dtype=np.float32)
    arr = (arr - arr.min()) / max(1e-8, arr.max() - arr.min())
    blobs = blob_log(arr, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    points = [(int(round(b[1])), int(round(b[0]))) for b in blobs]
    return points

# -------------------- Session state --------------------
for k in ["groups","all_points","last_file","disp_width","C_cache","last_M_hash","history"]:
    if k not in st.session_state:
        if k in ["groups","all_points","history"]:
            st.session_state[k] = []
        elif k=="disp_width":
            st.session_state[k] = 1000
        else:
            st.session_state[k] = None

# -------------------- UI --------------------
uploaded_file = st.file_uploader("Bild hochladen (jpg/png/tif)", type=["jpg","png","tif","tiff"])
if not uploaded_file:
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    st.session_state.groups = []
    st.session_state.all_points = []
    st.session_state.C_cache = None
    st.session_state.last_M_hash = None
    st.session_state.history = []
    st.session_state.last_file = uploaded_file.name

col1, col2 = st.columns([2,1])
with col2:
    st.sidebar.markdown("### Allgemeine Parameter")
    calib_radius = st.sidebar.slider("Kalibrier-Radius px",1,30,5)
    circle_radius = st.sidebar.slider("Marker-Radius px (Display)",1,12,5)
    dedup_dist_display = st.sidebar.slider("Min. Distanz Doppelzählung px",1,40,6)
    hema_default = st.sidebar.text_input("Hematoxylin vector","0.65,0.70,0.29")
    aec_default = st.sidebar.text_input("Chromogen vector","0.27,0.57,0.78")
    
    st.sidebar.markdown("### Blob-Detection Parameter")
    min_sigma = st.sidebar.slider("min_sigma", 1.0, 10.0, 2.0, 0.5)
    max_sigma = st.sidebar.slider("max_sigma", 1.0, 15.0, 6.0, 0.5)
    num_sigma = st.sidebar.slider("num_sigma", 1, 10, 5)
    threshold = st.sidebar.slider("threshold", 0.01, 0.2, 0.03, 0.005)
    
    try:
        hema_vec0 = np.array([float(x) for x in hema_default.split(",")])
        aec_vec0 = np.array([float(x) for x in aec_default.split(",")])
    except:
        hema_vec0 = np.array([0.65,0.70,0.29])
        aec_vec0 = np.array([0.27,0.57,0.78])

with col1:
    DISPLAY_WIDTH = st.slider("Anzeige-Breite px",300,1600,st.session_state.disp_width)
    st.session_state.disp_width = DISPLAY_WIDTH

# -------------------- Images --------------------
image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / float(W_orig)
H_disp = int(round(H_orig*scale))
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,H_disp),interpolation=cv2.INTER_AREA)
dedup_dist_orig = dedup_dist_display / scale

# -------------------- Click Handling --------------------
coords = streamlit_image_coordinates(Image.fromarray(image_disp),
                                     key=f"clickable_image_{st.session_state.last_file}",
                                     width=DISPLAY_WIDTH)

mode = st.sidebar.radio("Aktion", ["Kalibriere und zähle Gruppe (Klick)", "Punkt löschen", "Undo letzte Aktion"])
st.sidebar.markdown("---")
if st.sidebar.button("Reset (Alle Gruppen)"):
    st.session_state.history.append(("reset", {
        "groups": st.session_state.groups.copy(),
        "all_points": st.session_state.all_points.copy()
    }))
    st.session_state.groups=[]
    st.session_state.all_points=[]
    st.session_state.C_cache=None
    st.success("Zurückgesetzt.")

if coords:
    x_disp, y_disp = int(coords["x"]), int(coords["y"])
    x_orig = int(round(x_disp / scale))
    y_orig = int(round(y_disp / scale))

    if mode=="Punkt löschen":
        removed=[]
        new_all=[]
        for p in st.session_state.all_points:
            if is_near(p,(x_orig,y_orig),dedup_dist_orig):
                removed.append(p)
            else:
                new_all.append(p)
        if removed:
            st.session_state.history.append(("delete_points",{"removed":removed}))
            st.session_state.all_points=new_all
            for g in st.session_state.groups:
                g["points"]=[p for p in g["points"] if not is_near(p,(x_orig,y_orig),dedup_dist_orig)]
            st.success(f"{len(removed)} Punkt(e) gelöscht.")
        else:
            st.info("Kein Punkt in der Nähe gefunden.")
    elif mode=="Undo letzte Aktion":
        if st.session_state.history:
            action,payload=st.session_state.history.pop()
            if action=="add_group":
                idx = payload["group_idx"]
                if 0<=idx<len(st.session_state.groups):
                    grp=st.session_state.groups.pop(idx)
                    for pt in grp["points"]:
                        st.session_state.all_points=[p for p in st.session_state.all_points if p!=pt]
                    st.success("Letzte Gruppen-Aktion rückgängig gemacht.")
                else:
                    st.warning("Letzte Aktion konnte nicht rückgängig gemacht werden.")
            elif action=="delete_points":
                removed=payload["removed"]
                st.session_state.all_points.extend(removed)
                st.session_state.groups.append({"vec":None,"points":removed,"color":(0,255,0)})
                st.success("Gelöschte Punkte wiederhergestellt (als neue Gruppe).")
            elif action=="reset":
                st.session_state.groups=payload["groups"]
                st.session_state.all_points=payload["all_points"]
                st.success("Reset rückgängig gemacht.")
            else:
                st.warning("Undo: unbekannte Aktion.")
        else:
            st.info("Keine Aktion zum Rückgängig machen.")
    else:
        patch = extract_patch(image_orig,x_orig,y_orig,calib_radius)
        vec = median_od_vector_from_patch(patch)
        if vec is None:
            st.warning("Patch unbrauchbar (zu homogen oder außerhalb).")
        else:
            M = make_stain_matrix(vec,hema_vec0)
            M_hash = tuple(np.round(M.flatten(),6).tolist())
            recompute=False
            if st.session_state.C_cache is None or st.session_state.last_M_hash!=M_hash:
                recompute=True
            if recompute:
                C_full=deconvolve(image_orig,M)
                st.session_state.C_cache=C_full
                st.session_state.last_M_hash=M_hash
            else:
                C_full=st.session_state.C_cache
            channel_full=C_full[:,:,0]
            points=detect_blobs(channel_full,
                                min_sigma=min_sigma,
                                max_sigma=max_sigma,
                                num_sigma=num_sigma,
                                threshold=threshold)
            new_points=dedup_new_points(points,st.session_state.all_points,min_dist=dedup_dist_orig)
            if new_points:
                color=(np.random.randint(50,230),np.random.randint(50,230),np.random.randint(50,230))
                group={"vec":vec.tolist(),"points":new_points,"color":color}
                st.session_state.history.append(("add_group",{"group_idx":len(st.session_state.groups)}))
                st.session_state.groups.append(group)
                st.session_state.all_points.extend(new_points)
                st.success(f"Gruppe hinzugefügt — neue Kerne: {len(new_points)}")
            else:
                st.info("Keine neuen Kerne gefunden.")

# -------------------- Anzeige & Export --------------------
display_canvas=image_disp.copy()
for i,g in enumerate(st.session_state.groups):
    col=g.get("color",(0,255,0))
    for (x,y) in g["points"]:
        x_disp=int(round(x*scale))
        y_disp=int(round(y*scale))
        cv2.circle(display_canvas,(x_disp,y_disp),circle_radius,col,-1)
st.image(display_canvas,use_column_width=True)

st.markdown("## Ergebnisse")
st.write(f"🔹 Gruppen gesamt: {len(st.session_state.groups)}")
for i,g in enumerate(st.session_state.groups):
    st.write(f"• Gruppe {i+1}: {len(g['points'])} neue Kerne")
st.markdown(f"**Gesamt (unique Kerne): {len(st.session_state.all_points)}**")

# CSV Export
if st.session_state.all_points:
    rows=[]
    for i,g in enumerate(st.session_state.groups):
        for (x,y) in g["points"]:
            rows.append({"Group":i+1,"X_original":x,"Y_original":y,"X_display":int(round(x*scale)),"Y_display":int(round(y*scale))})
    df=pd.DataFrame(rows)
    st.download_button("📥 CSV exportieren (Gruppen)",df.to_csv(index=False).encode("utf-8"),
                       file_name="kern_gruppen_blobs.csv",mime="text/csv")
    df_unique=pd.DataFrame(st.session_state.all_points,columns=["X_original","Y_original"])
    df_unique["X_display"]=(df_unique["X_original"]*scale).round().astype(int)
    df_unique["Y_display"]=(df_unique["Y_original"]*scale).round().astype(int)
    st.download_button("📥 CSV exportieren (unique)",df_unique.to_csv(index=False).encode("utf-8"),
                       file_name="kern_unique_blobs.csv",mime="text/csv")
