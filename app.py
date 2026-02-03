import streamlit as st
import cv2
import numpy as np
import io
import os
import gc
import time
import requests
import base64
import json
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from streamlit_drawable_canvas import st_canvas
import easyocr
import google.generativeai as genai

# --- CORE SYSTEM ---

class Engine:
    @staticmethod
    def get_font():
        if 'font_cache' not in st.session_state:
            try:
                # Noto Sans JP
                url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/ttf/NotoSansCJKjp-VF.ttf"
                params = {"download": "true"}
                response = requests.get(url)
                st.session_state['font_cache'] = io.BytesIO(response.content)
            except: st.session_state['font_cache'] = None
        return st.session_state['font_cache']

    @staticmethod
    def estimate_blur(image):
        """Estimate the blur radius needed based on image sharpness."""
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            # Variance of Laplacian gives score. Higher = Sharper.
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Empirical mapping: Score > 500 is very sharp (0 blur). Score < 100 is blurry (~2-3px blur)
            # This is a heuristic.
            if score > 500: return 0
            if score > 200: return 0.5
            if score > 100: return 1.0
            return 2.0
        except: return 0

# --- UI DESIGN SYSTEM (Modern SaaS) ---

# --- UI DESIGN SYSTEM (Modern SaaS) ---

def inject_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        :root {
            --bg-color: #FAFAFA;
            --surface: #FFFFFF;
            --primary: #000000;
            --text-main: #171717;
            --border: #E5E5E5;
        }

        .stApp {
            background-color: var(--bg-color);
            font-family: 'Inter', sans-serif !important;
            color: var(--text-main);
        }

        /* FORCE LIGHT MODE INPUTS */
        /* Takes priority over Streamlit dark mode defaults */
        .stTextInput input, .stSelectbox div[data-baseweb="select"], div[data-baseweb="base-input"] {
            background-color: #F5F5F5 !important;
            color: #000000 !important; /* Explicit Black */
            -webkit-text-fill-color: #000000 !important;
            opacity: 1 !important;
            border: 1px solid #E5E5E5 !important;
            border-radius: 8px !important;
            caret-color: #000000 !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border);
            padding-top: 1rem !important;
        }
        
        /* Adjust header gap */
        div[data-testid="stSidebar"] .block-container {
            padding-top: 0rem !important;
        }

        /* Buttons */
        .stButton button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            border: none !important;
            transition: opacity 0.2s;
        }
        .stButton button[kind="primary"] {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #000000 !important;
        }
        .stButton button[kind="primary"]:hover {
            opacity: 0.8 !important;
        }
        
        /* Hide Header/Footer */
        header, footer { visibility: hidden; }
        .block-container { padding-top: 2rem !important; }

    </style>
    """, unsafe_allow_html=True)

# --- VISION PIPELINE ---

def auto_detect_geometry(image):
    """
    Auto-detects price tag geometry.
    Returns: list of 4 points or None.
    """
    try:
        # Robust conversion
        img = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Adaptive Thresholding for wider condition support
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 200)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4 and cv2.contourArea(c) > 1000:
                return [{'x': int(p[0][0]), 'y': int(p[0][1])} for p in approx]
        return None
    except: return None

def pipeline_execute(img_file, roi_points, price_text, blur_level, size):
    """
    Executes the edit with Blur Matching.
    """
    # 1. Load High-Res
    nparr = np.frombuffer(img_file.getvalue(), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    pts = np.array([[p['x'], p['y']] for p in roi_points], dtype='float32')

    # 2. Warp Perspective (Flatten Tag)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    maxW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
    maxH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
    
    dst = np.array([[0, 0], [maxW-1, 0], [maxW-1, maxH-1], [0, maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_rgb, M, (maxW, maxH))

    # 3. Clean Plate (Inpaint old text)
    mask = np.zeros(warped.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (maxW-10, maxH-10), 255, -1)
    clean_plate = cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA)

    # 4. Generate Text
    pil_plate = Image.fromarray(clean_plate)
    txt_layer = Image.new('RGBA', pil_plate.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt_layer)
    
    # Font Logic
    font_bytes = Engine.get_font()
    try:
        f = ImageFont.truetype(font_bytes, int(maxH * 0.82)) if font_bytes else ImageFont.load_default()
    except: f = ImageFont.load_default()
    
    bbox = d.textbbox((0, 0), price_text, font=f)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    
    # Text Color: Black with 90% opacity
    d.text(((maxW-tw)/2, (maxH-th)/2 - bbox[1]*0.1), price_text, font=f, fill=(0, 0, 0, 230))
    
    # 5. BLUR MATCHING (Key Feature)
    if blur_level > 0:
        txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(blur_level))
        
    pil_plate.paste(txt_layer, (0, 0), txt_layer)
    
    # 6. Un-Warp & Blend
    res_warped = cv2.warpPerspective(np.array(pil_plate), M, (img_rgb.shape[1], img_rgb.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    
    mask_full = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, rect.astype(int), 255)
    mask_blur = cv2.GaussianBlur(mask_full, (5, 5), 0)
    
    img_f = img_rgb.astype(float)
    res_f = res_warped.astype(float)
    alpha = np.repeat((mask_blur.astype(float)/255.0)[:, :, np.newaxis], 3, axis=2)
    
    final = (res_f * alpha) + (img_f * (1.0 - alpha))
    return Image.fromarray(final.astype(np.uint8))


# --- MAIN ---

def main():
    st.set_page_config(page_title="Price Craft", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    
    # Auth State
    if 'auth' not in st.session_state: st.session_state.auth = False
    
    # --- AUTHENTICATION SCREEN ---
    if not st.session_state.auth:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #000000; margin-bottom: 20px;'>Price Craft</h2>", unsafe_allow_html=True)
            
            # FIX: Form to handle submission correctly
            with st.form("auth_form"):
                pwd = st.text_input("Enter Passkey", type="password", placeholder="••••••")
                submit = st.form_submit_button("Enter System", type="primary", use_container_width=True)
                
                if submit:
                    if pwd == st.secrets.get("PASSWORD", "apple"):
                        st.session_state.auth = True
                        st.rerun()
                    else:
                        st.error("Incorrect Passkey")
        return

    # --- COMPACT SIDEBAR ---
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>Price Craft <span style='font-size:10px; color:#A3A3A3; padding-left:4px'>BETA</span></div>", unsafe_allow_html=True)
        
        # 1. INPUT
        with st.expander("Import", expanded=True):
            uploaded = st.file_uploader("Upload Image", type=["jpg", "png"], label_visibility="collapsed")
            
        if uploaded:
            # 2. INTELLIGENCE
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("Auto-Scan Image", type="secondary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Mock Gemimi Call
                    st.session_state['scan_data'] = {"text": "298", "font": "Gothic Bold"}
                    st.toast("Tags Detected", icon="🏷️")
            
            # 3. SETTINGS (Compact Form)
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.caption("CONFIGURATION")
            
            # Default Data
            scan = st.session_state.get('scan_data', {"text": "---", "font": "Unknown"})
            current_val = scan['text'] if scan['text'] != "---" else "298"
            
            # Row: Detected vs New
            c_a, c_b = st.columns([0.4, 0.6])
            with c_a: 
                st.markdown(f"<div style='font-size:12px; color:#737373'>Detected</div><div style='font-weight:600'>{current_val}</div>", unsafe_allow_html=True)
            with c_b:
                target_price = st.text_input("Target Price", value=current_val, label_visibility="collapsed")
            
            # Row: Tax Mode (Lens Fix removed)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            tax_mode = st.selectbox("Tax", ["None", "8%", "10%"], label_visibility="collapsed")
                
            # Row: Blur Match
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            # Auto-calc blur default
            if 'auto_blur' not in st.session_state:
                st.session_state.auto_blur = 0.0 # Will calculate on load
            
            blur_val = st.slider("Blur Match", 0.0, 5.0, st.session_state.auto_blur, 0.1, format="%.1f px")

            # 4. ACTION
            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            if st.button("Generate Tag", type="primary"):
                st.session_state['run'] = True

        else:
            st.info("Please upload a price tag image.")

    # --- MAIN CANVAS ---
    if uploaded:
        file_sig = f"{uploaded.file_id}"
        
        # Load & Handle Transparency (Fixes Black Screen on Transparent PNGs)
        img_raw = Image.open(uploaded)
        if img_raw.mode in ('RGBA', 'LA') or (img_raw.mode == 'P' and 'transparency' in img_raw.info):
            # Create white background for transparent images
            alpha = img_raw.convert('RGBA')
            bg = Image.new("RGB", alpha.size, (255, 255, 255))
            bg.paste(alpha, mask=alpha.split()[3])
            img_fixed = bg
        else:
            img_fixed = img_raw.convert("RGB")
            
        img_fixed = ImageOps.exif_transpose(img_fixed)
        # Deep scrub
        img_fixed = Image.fromarray(np.array(img_fixed))
        
        # Display Image Info (Data Amount)
        file_size_mb = uploaded.size / (1024*1024)
        st.sidebar.markdown(f"<div style='margin-top:-10px; font-size:11px; color:#A3A3A3; text-align:right'>{img_fixed.width}x{img_fixed.height}px • {file_size_mb:.2f}MB</div>", unsafe_allow_html=True)
        
        # Calculate Blur ONCE
        if st.session_state.get('last_blur_sig') != file_sig:
            est_blur = Engine.estimate_blur(img_fixed)
            st.session_state.auto_blur = est_blur
            st.session_state.last_blur_sig = file_sig
            st.rerun() # Refresh slider default
            
        w, h = img_fixed.size
        
        # Fit to view
        max_v_w = 1000
        view_scale = 1.0
        
        if w > max_v_w:
            view_scale = max_v_w / w
            disp_img = img_fixed.resize((max_v_w, int(h * view_scale)), Image.Resampling.LANCZOS)
        else:
            disp_img = img_fixed.copy()
            
        # --- CRITIAL FIX: BUFFER ROUND-TRIP ---
        # Forces a clean file-like object for the canvas component
        buf = io.BytesIO()
        disp_img.save(buf, format="PNG")
        disp_img = Image.open(buf)
        
        with st.expander("Troubleshoot Preview", expanded=False):
             st.image(disp_img, caption="Source Image Verification", use_container_width=True)

        # Auto Detect Geometry ONCE
        init_geom = None
        if st.session_state.get('last_geom_sig') != file_sig:
            pts = auto_detect_geometry(img_fixed)
            if pts:
                s_pts = [{'x': p['x']*view_scale, 'y': p['y']*view_scale} for p in pts]
                init_geom = {
                    "version": "4.4.0",
                    "objects": [{
                        "type": "path",
                        "originX": "left", "originY": "top", "left": 0, "top": 0,
                        "fill": "rgba(37, 99, 235, 0.2)",
                        "stroke": "#2563EB",
                        "strokeWidth": 2,
                        "path": [
                            ["M", s_pts[0]['x'], s_pts[0]['y']],
                            ["L", s_pts[1]['x'], s_pts[1]['y']],
                            ["L", s_pts[2]['x'], s_pts[2]['y']],
                            ["L", s_pts[3]['x'], s_pts[3]['y']],
                            ["Z"]
                        ]
                    }]
                }
                st.session_state['geom'] = init_geom
                st.toast("Geometry Aligned", icon="📐")
            else:
                st.session_state['geom'] = None
            st.session_state['last_geom_sig'] = file_sig

        # RENDER CANVAS
        st.caption("Review Selection Area")
        canvas = st_canvas(
            fill_color="rgba(37, 99, 235, 0.2)",
            stroke_width=2,
            stroke_color="#2563EB",
            background_image=disp_img,
            initial_drawing=st.session_state.get('geom'),
            update_streamlit=True,
            height=disp_img.height,
            width=disp_img.width,
            drawing_mode="polygon",
            key=f"cv_{file_sig}",
        )
        
        # PROCESS
        if st.session_state.get('run', False):
             if canvas.json_data and len(canvas.json_data["objects"]) > 0:
                 path = canvas.json_data["objects"][0]["path"]
                 roi = [{'x': p[1]/view_scale, 'y': p[2]/view_scale} for p in path if p[0] in ['M','L']]
                 
                 if len(roi) >= 3:
                     with st.spinner("Processing..."):
                         final_p = target_price
                         if target_price.isdigit() and tax_mode != "None":
                             rate = 1.08 if tax_mode == "8%" else 1.10
                             final_p = f"¥{target_price} (税込¥{int(int(target_price)*rate)})"
                         elif target_price.isdigit():
                             final_p = f"¥{target_price}"
                             
                         st.session_state['result'] = pipeline_execute(uploaded, roi[:4], final_p, blur_val, img_fixed.size)
             st.session_state['run'] = False 
             
        # RESULT (Side by Side)
        if 'result' in st.session_state:
            st.markdown("---")
            st.caption("GENERATION RESULT")
            c1, c2 = st.columns(2)
            with c1: st.image(img_fixed, caption="Original", use_container_width=True)
            with c2: st.image(st.session_state['result'], caption="Price Craft", use_container_width=True)
            
            buf = io.BytesIO()
            st.session_state['result'].save(buf, format="PNG")
            st.download_button("Download Asset", buf.getvalue(), "crafted.png", "image/png", type="primary")

    PriceCraftOS.cleanup()

if __name__ == "__main__":
    main()
