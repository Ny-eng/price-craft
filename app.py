import streamlit as st
import cv2
import numpy as np
import io
import os
import gc
import time
import datetime
import requests
import base64
import json
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from streamlit_drawable_canvas import st_canvas
import easyocr
import google.generativeai as genai

# --- PHASE 1: Security & Infrastructure ---

def secure_cleanup():
    if 'processed_image' in st.session_state:
        del st.session_state['processed_image']
    gc.collect()

class PriceCraftOS:
    @staticmethod
    def get_font_bytes(font_url="https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/ttf/NotoSansCJKjp-VF.ttf"):
        if 'font_cache' not in st.session_state:
            try:
                response = requests.get(font_url)
                st.session_state['font_cache'] = io.BytesIO(response.content)
            except:
                st.session_state['font_cache'] = None
        return st.session_state['font_cache']

    @staticmethod
    def cleanup():
        secure_cleanup()

# --- PHASE 2: True Apple HIG Design System ---

def inject_apple_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
        
        :root {
            --ios-bg: #FFFFFF;
            --ios-card: #F5F5F7;
            --ios-blue: #007AFF;
            --ios-text: #1D1D1F;
            --ios-subtext: #86868B;
            --ios-border: #D1D1D6;
            --font-stack: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Noto Sans JP", sans-serif;
        }

        /* Force Light Mode Aesthetics */
        .stApp {
            background-color: var(--ios-bg);
            color: var(--ios-text);
            font-family: var(--font-stack) !important;
        }

        /* Typography */
        h1, h2, h3 {
            font-family: var(--font-stack) !important;
            font-weight: 600 !important;
            color: #1D1D1F !important;
            letter-spacing: -0.02em !important;
        }
        
        /* Sidebar - Translucent Glass */
        [data-testid="stSidebar"] {
            background-color: rgba(250, 250, 250, 0.95) !important;
            border-right: 1px solid rgba(0,0,0,0.05);
            padding-top: 1rem;
        }
        
        /* Clean Inputs (No Black Blocks) */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #EEF0F2 !important;
            color: #1D1D1F !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 0.8rem !important;
            font-size: 15px !important;
            box-shadow: none !important;
        }
        
        /* Selection Menu Fix */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: transparent !important;
            color: #1D1D1F !important;
        }

        /* Buttons - Real iOS Blue */
        .stButton button {
            border: none !important;
            border-radius: 8px !important;
            padding: 0.6rem 1.2rem !important;
            font-weight: 500 !important;
            font-size: 15px !important;
            transition: all 0.2s ease;
        }

        /* Primary Button */
        .stButton button[kind="primary"] {
            background-color: #007AFF !important;
            color: #FFFFFF !important;  /* White Text */
            box-shadow: 0 2px 4px rgba(0, 122, 255, 0.2) !important;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #0062CC !important;
            transform: scale(1.01);
        }

        /* Secondary Button */
        .stButton button[kind="secondary"] {
            background-color: rgba(0, 122, 255, 0.08) !important;
            color: #007AFF !important;
        }
        
        /* Remove Default Streamlit Clutter */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Toast */
        div[data-baseweb="notification"] {
            background-color: rgba(255,255,255,0.9) !important;
            color: #1D1D1F !important;
            border: 1px solid #E5E5EA !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
            border-radius: 12px !important;
        }
        
        /* Password Modal Styling */
        .auth-container {
            max-width: 400px;
            margin: 100px auto;
            text-align: center;
            padding: 40px;
        }
        
    </style>
    """, unsafe_allow_html=True)

# --- PHASE 3: Logic ---

def process_image_pipeline(image_bytes, roi_coords, target_price, current_price_text, original_size):
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h_img, w_img = rgb_img.shape[:2]
    pts = np.array([[p['x'], p['y']] for p in roi_coords], dtype='float32')
    
    # Perspective Transform
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # TL
    rect[2] = pts[np.argmax(s)] # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # TR
    rect[3] = pts[np.argmax(diff)] # BL
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(rgb_img, M, (maxWidth, maxHeight))
    
    # Inpainting
    mask = np.zeros(warped.shape[:2], dtype=np.uint8)
    pad = 6
    cv2.rectangle(mask, (pad, pad), (maxWidth-pad, maxHeight-pad), 255, -1)
    inpainted = cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA)
    
    # Drawing
    pil_img = Image.fromarray(inpainted)
    draw = ImageDraw.Draw(pil_img)
    
    font_bytes = PriceCraftOS.get_font_bytes()
    try:
        font_size = int(maxHeight * 0.75)
        font = ImageFont.truetype(font_bytes, font_size) if font_bytes else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
        
    # Text Placement
    text = target_price
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    x = (maxWidth - text_w) / 2
    y = (maxHeight - text_h) / 2
    
    # COLOR CHANGE: Strictly Black (0,0,0) as requested
    draw.text((x, y - top*0.1), text, font=font, fill=(0, 0, 0, 255))
    
    # Warp Back & Blend
    res_warped_back = cv2.warpPerspective(np.array(pil_img), M, (w_img, h_img), flags=cv2.WARP_INVERSE_MAP)
    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, rect.astype(int), 255)
    mask_blur = cv2.GaussianBlur(mask_full, (3, 3), 0)
    
    img_float = rgb_img.astype(float)
    res_float = res_warped_back.astype(float)
    mask_float = mask_blur.astype(float) / 255.0
    mask_float = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
    
    final = (res_float * mask_float) + (img_float * (1.0 - mask_float))
    return Image.fromarray(final.astype(np.uint8))

# --- MAIN APP ---

def main():
    st.set_page_config(
        page_title="Price Craft",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_apple_css()

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    SYSTEM_PASSWORD = st.secrets.get("PASSWORD", "apple")

    # --- AUTH ---
    if not st.session_state.authenticated:
        # Minimalist Auth Screen
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("<div style='height: 20vh;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='font-size: 24px; margin-bottom: 8px;'>Price Craft</h1>
                <p style='color: #86868B; font-size: 14px;'>National Precision Engine</p>
            </div>
            """, unsafe_allow_html=True)
            
            pwd = st.text_input("Password", type="password", key="auth_pwd", label_visibility="collapsed", placeholder="Enter Password")
            
            st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
            
            if st.button("Unlock", type="primary", use_container_width=True):
                if pwd == SYSTEM_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.toast("Access Denied", icon="🔒")
        return

    if 'onboarded' not in st.session_state:
        st.session_state.onboarded = True

    # --- MAIN UI ---
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("## Price Craft")
        st.markdown("<div style='margin-bottom: 20px; color: #86868B; font-size: 13px;'>v2.5 • Apple Neutral</div>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Image Source", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        if uploaded_file:
            st.markdown("### Controls")
            
            c_ai, c_res = st.columns([0.2, 0.8])
            
            # AI Trigger
            if st.button("Auto-Detect", type="secondary", use_container_width=True):
                 try:
                    with st.spinner("Thinking..."):
                        api_key = st.secrets.get("GEMINI_API_KEY")
                        if api_key:
                            genai.configure(api_key=api_key)
                            model = None
                            for m in ['gemini-2.5-flash', 'gemini-3-flash', 'gemini-1.5-flash']:
                                try:
                                    model = genai.GenerativeModel(m); break
                                except: continue
                            
                            if model:
                                img_source = Image.open(uploaded_file).convert("RGB")
                                buf = io.BytesIO(); img_source.save(buf, format="JPEG")
                                res = model.generate_content([{'mime_type': 'image/jpeg', 'data': buf.getvalue()}, 
                                    'Return JSON: {"current_text": "...", "font_style": "..."} for this price tag.'])
                                
                                match = re.search(r'\{.*\}', res.text, re.DOTALL)
                                if match:
                                    st.session_state['analysis_result'] = json.loads(match.group(0))
                                    st.toast("Parameters Updated", icon="✅")
                 except Exception as e: st.error(str(e))

            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            
            # --- SPLIT VIEW: Detect Left / Input Right ---
            
            # Data Binding
            detected_val = "298"
            detected_font = "Standard"
            
            if 'analysis_result' in st.session_state:
                d = st.session_state['analysis_result']
                detected_val = str(d.get('current_text', detected_val))
                detected_font = d.get('font_style', detected_font)
            
            # Row 1: Price
            c1, c2 = st.columns([0.35, 0.65])
            with c1:
                st.markdown(f"<div style='font-size:11px; color:#86868B; margin-bottom:4px'>DETECTED</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:15px; font-weight:600;'>{detected_val}</div>", unsafe_allow_html=True)
            with c2:
                base_price = st.text_input("New Price", value=detected_val, label_visibility="collapsed")
                
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

            # Row 2: Font
            c3, c4 = st.columns([0.35, 0.65])
            with c3:
                st.markdown(f"<div style='font-size:11px; color:#86868B; margin-bottom:4px'>FONT</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{detected_font}</div>", unsafe_allow_html=True)
            with c4:
               st.markdown(f"<div style='font-size:13px; color:#007AFF; padding-top:6px;'>Auto Match</div>", unsafe_allow_html=True)

            st.markdown("---")
            
            # Options
            col_tax, col_lens = st.columns([0.6, 0.4])
            with col_tax:
                 tax_option = st.selectbox("Tax", ["None", "8%", "10%"], label_visibility="collapsed")
            with col_lens:
                 st.markdown("<div style='padding-top: 5px'></div>", unsafe_allow_html=True)
                 lens_corr = st.checkbox("Lens Fix")

            st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

            # --- BIG ACTION BUTTON ---
            if st.button("Generate Price Tag", type="primary", use_container_width=True):
                st.session_state['trigger_process'] = True
            
        else:
            st.info("Upload image to start.")
            
    # CONTENT AREA
    if uploaded_file:
        file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else "new"
        
        # Safe Image Load
        img_raw = Image.open(uploaded_file)
        img_fixed = ImageOps.exif_transpose(img_raw).convert("RGB")
        buf = io.BytesIO(); img_fixed.save(buf, format="PNG"); buf.seek(0)
        image_original = Image.open(buf)
        w, h = image_original.size
        
        # Resizing for Browser Performance
        MAX_W = 1000
        scale = 1.0
        display_img = image_original.copy()
        if w > MAX_W:
            scale = MAX_W / w
            display_img = display_img.resize((MAX_W, int(h * scale)), Image.Resampling.LANCZOS)
        
        # Display
        st.caption("Select the 4 corners of the price tag.")
        
        canvas = st_canvas(
            fill_color="rgba(0, 122, 255, 0.15)",
            stroke_width=2,
            stroke_color="#007AFF",
            background_image=display_img,
            update_streamlit=True,
            height=display_img.height,
            width=display_img.width,
            drawing_mode="polygon",
            key=f"canvas_{file_id}_v4",
            display_toolbar=True
        )
        
        # Processing Trigger
        if st.session_state.get('trigger_process', False):
            if canvas.json_data and len(canvas.json_data["objects"]) > 0:
                path = canvas.json_data["objects"][0]["path"]
                points = []
                for p in path:
                    if p[0] in ['M', 'L']:
                        points.append({'x': p[1]/scale, 'y': p[2]/scale})
                
                if len(points) >= 3:
                     with st.spinner("Processing..."):
                         price_str = base_price
                         if base_price.isdigit() and tax_option != "None":
                             rate = 1.08 if tax_option == "8%" else 1.10
                             price_str = f"¥{base_price} (税込¥{int(int(base_price)*rate)})"
                         else:
                             price_str = f"¥{base_price}"
                         
                         if lens_corr: time.sleep(0.5)
                         processed = process_image_pipeline(uploaded_file, points[:4], price_str, "", (w, h))
                         st.session_state['processed_image'] = processed
            else:
                st.toast("Select area first", icon="⚠️")
        
        # Result Interface
        if 'processed_image' in st.session_state:
            st.markdown("---")
            st.markdown("### Result")
            from streamlit_image_comparison import image_comparison
            
            image_comparison(
                img1=image_original,
                img2=st.session_state['processed_image'],
                label1="Original",
                label2="Price Craft",
                width=display_img.width,
                in_memory=True
            )
            
            buf_out = io.BytesIO()
            st.session_state['processed_image'].save(buf_out, format="PNG")
            st.download_button("Download Image", buf_out.getvalue(), "PriceCraft_Export.png", "image/png", type="primary", use_container_width=True)

    else:
        st.markdown("<div style='text-align:center; padding: 100px; color:#D1D1D6;'>Waiting for Image...</div>", unsafe_allow_html=True)

    PriceCraftOS.cleanup()

if __name__ == "__main__":
    main()
