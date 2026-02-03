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
            border: 1px solid #E5E5E5 !important;
            border-radius: 8px !important;
            caret-color: #000000 !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border);
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

# ... (Vision Pipeline remains unchanged) ...
# (We assume helper classes Engine and functions auto_detect_geometry, pipeline_execute exist as before. 
# but the replace_file_content tool requires context. I will target the styles and main function areas)

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
            
            # Row: Tax & Features
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            t1, t2 = st.columns(2)
            with t1:
                tax_mode = st.selectbox("Tax", ["None", "8%", "10%"], label_visibility="collapsed")
            with t2:
                lens_fix = st.checkbox("Lens Fix", value=True)
                
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
        
        # Load & optimize
        img_raw = Image.open(uploaded).convert("RGB")
        img_fixed = ImageOps.exif_transpose(img_raw)
        
        # Calculate Blur ONCE
        if st.session_state.get('last_blur_sig') != file_sig:
            est_blur = Engine.estimate_blur(img_fixed)
            st.session_state.auto_blur = est_blur
            st.session_state.last_blur_sig = file_sig
            st.rerun() # Refresh slider default
            
        w, h = img_fixed.size
        
        # Fit to view
        max_v_w = 1200
        view_scale = 1.0
        disp_img = img_fixed
        if w > max_v_w:
            view_scale = max_v_w / w
            disp_img = img_fixed.resize((max_v_w, int(h * view_scale)))

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
