import streamlit as st
import cv2
import numpy as np
import io
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import easyocr

# --- CONSTANTS & CONFIG ---
PAGE_TITLE = "Price Craft Studio"
PAGE_ICON = "🏷️"

# --- CORE LOGIC ---

class VisionEngine:
    @staticmethod
    @st.cache_resource
    def get_reader():
        """Initialize EasyOCR Reader (Cached)."""
        # Loading English and Japanese
        return easyocr.Reader(['ja', 'en'], gpu=False)

    @staticmethod
    def get_font():
        """Cache remote font for Japanese support."""
        if 'font_cache' not in st.session_state:
            try:
                url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/ttf/NotoSansCJKjp-VF.ttf"
                r = requests.get(url, timeout=5)
                st.session_state['font_cache'] = io.BytesIO(r.content)
            except: 
                st.session_state['font_cache'] = None
        return st.session_state['font_cache']

    @staticmethod
    @st.cache_data(show_spinner=False)
    def scan_all_text(image_bytes):
        """Scans all text in the image using EasyOCR with optimized params for price tags."""
        try:
            reader = VisionEngine.get_reader()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_np = np.array(image)
            
            # Run OCR with Tuned Parameters for Price Tags (Big Numbers)
            results = reader.readtext(
                img_np,
                detail=1,
                mag_ratio=1.5,       # Enlarge image to see text better
                text_threshold=0.5,  # Lower threshold to catch colored text
                low_text=0.3,        # Catch lower confidence text
                contrast_ths=0.1,    # Text/Background contrast threshold
                adjust_contrast=0.5  # Boost contrast before detection
            )
            
            # Format results
            detected_items = []
            for (bbox, text, prob) in results:
                # Filter noise
                if prob > 0.2 and text.strip():
                    detected_items.append({
                        'id': f"{text}_{bbox[0][0]}", 
                        'text': text,
                        'bbox': bbox,
                        'prob': prob
                    })
            
            # Sort by Y position (Top to Bottom)
            detected_items.sort(key=lambda x: x['bbox'][0][1])
            return detected_items
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return []

    @staticmethod
    def replace_text_items(image_bytes, items_to_modify, blur_rad=0.0):
        """Replaces specific text items in the image."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        
        # 1. Inpaint (Remove old text)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        
        for item in items_to_modify:
            box = np.array(item['bbox'], dtype=np.int32)
            cv2.fillConvexPoly(mask, box, 255)
            
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3) # Stronger dilation
        
        clean_bg = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
        
        # 2. Draw New Text
        pil_img = Image.fromarray(clean_bg)
        txt_layer = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        font_bytes = VisionEngine.get_font()
        
        for item in items_to_modify:
            new_text = item['new_text']
            if not new_text: continue
            
            box = item['bbox'] 
            tl, tr, br, bl = box
            height = int(abs(bl[1] - tl[1]))
            
            # Determine Color (Simple Heuristic: Use avg color of surrounding or default black)
            # Defaulting to High-Contrast Dark Gray/Black for Prices
            text_color = (10, 10, 10, 240)
            
            try:
                f_size = int(height * 0.85)
                if f_size < 12: f_size = 12
                font = ImageFont.truetype(font_bytes, f_size) if font_bytes else ImageFont.load_default()
            except: font = ImageFont.load_default()
            
            text_bbox = draw.textbbox((0, 0), new_text, font=font)
            t_w = text_bbox[2] - text_bbox[0]
            t_h = text_bbox[3] - text_bbox[1]
            
            center_x = (tl[0] + br[0]) / 2
            center_y = (tl[1] + br[1]) / 2
            
            pos_x = center_x - (t_w / 2)
            pos_y = center_y - (t_h / 2) - (text_bbox[1] * 0.1)
            
            draw.text((pos_x, pos_y), new_text, font=font, fill=text_color)

        # 3. Blur Match
        if blur_rad > 0:
            txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(blur_rad))
            
        pil_img.paste(txt_layer, (0,0), txt_layer)
        return pil_img

# --- UI ---

def inject_premium_css():
    st.markdown("""
    <style>
        /* FORCE LIGHT THEME & RESET */
        :root {
            --primary-color: #000000;
            --background-color: #ffffff;
            --secondary-background-color: #f8f9fa;
            --text-color: #000000;
            --font: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* App Background */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6, .stMarkdown h3 { 
            color: #000000 !important; 
            font-weight: 700 !important;
        }
        
        /* Inputs - High Visibility */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            border: 2px solid #e5e5e5 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }
        .stTextInput input:focus {
            border-color: #000000 !important;
        }
        
        /* Disabled Inputs (Detected Text) */
        .stTextInput input:disabled {
            background-color: #f1f3f5 !important;
            color: #333333 !important;
            -webkit-text-fill-color: #333333 !important;
            border-color: #f1f3f5 !important;
        }
        
        /* Buttons */
        button[kind="primary"] {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: none !important;
            font-weight: 600 !important;
        }
        
        /* Tables/Lists */
        .element-container { color: #000 !important; }
        
        /* Layout */
        .block-container { 
            padding-top: 2rem !important; 
            padding-bottom: 3rem !important;
            max-width: 1400px !important;
        }
        header, footer { visibility: hidden; }
        
        /* Custom Labels */
        label { color: #444 !important; font-size: 13px !important; font-weight: 600 !important; }
        
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    inject_premium_css()
    
    # --- AUTH ---
    if 'auth' not in st.session_state: st.session_state.auth = False
    if not st.session_state.auth:
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            st.markdown("<br><br><h2 style='text-align:center'>Price Craft Studio</h2>", unsafe_allow_html=True)
            with st.form("login"):
                pwd = st.text_input("Access Key", type="password")
                submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                if submit:
                    if pwd == st.secrets.get("PASSWORD", "apple"):
                        st.session_state.auth = True
                        st.rerun()
                    else: st.error("Invalid Key")
        return

    # --- STATE ---
    if 'ocr_results' not in st.session_state: st.session_state.ocr_results = []
    
    # --- HEADER ---
    st.markdown("<h3 style='margin-bottom:20px'>Price Craft <span style='color:#999; font-weight:400'>Studio</span></h3>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=['png','jpg','jpeg'], label_visibility="collapsed")
    
    if uploaded:
        # File Handling: Read bytes once to share between EasyOCR/PIL without seeking issues
        file_bytes = uploaded.getvalue()
        
        # Display Logic
        img_pil = Image.open(io.BytesIO(file_bytes))
        # Handle alpha
        if img_pil.mode in ('RGBA', 'LA') or (img_pil.mode == 'P' and 'transparency' in img_pil.info):
            bg = Image.new("RGB", img_pil.size, (255, 255, 255))
            bg.paste(img_pil, mask=img_pil.convert('RGBA').split()[3])
            img_pil = bg
        else:
            img_pil = img_pil.convert("RGB")
        img_fix = ImageOps.exif_transpose(img_pil)

        c_left, c_right = st.columns([0.4, 0.6], gap="large")
        
        # --- LEFT PANEL: CONTROL ---
        with c_left:
            st.markdown("#### 1. Analyze")
            if st.button("🔍 Scan Text", type="primary", use_container_width=True):
                with st.spinner("Scanning image... (This may take a moment)"):
                    # Use the raw bytes_io for Thread safety in OCR
                    st.session_state.ocr_results = VisionEngine.scan_all_text(file_bytes)
                    if not st.session_state.ocr_results:
                        st.warning("No text detected. Try a clearer image.")
            
            # SETTINGS
            with st.expander("Settings", expanded=False):
                blur_v = st.slider("Blur Intensity", 0.0, 5.0, 0.5, 0.1)

            # TEXT EDITOR
            st.markdown("#### 2. Edit Text")
            items_to_modify = []
            
            if st.session_state.ocr_results:
                with st.container(height=600):
                    for i, item in enumerate(st.session_state.ocr_results):
                        # Use a clean 3-col layout
                        c1, c2, c3 = st.columns([0.15, 0.35, 0.5])
                        with c1:
                            # Show confidence as tiny text
                            st.caption(f"{int(item['prob']*100)}%")
                        with c2:
                            # Original text (Disabled Input)
                            st.text_input(f"org_{i}", value=item['text'], disabled=True, label_visibility="collapsed", key=f"dis_{item['id']}")
                        with c3:
                            # New Input
                            new_val = st.text_input(
                                f"new_{i}", 
                                placeholder="Change to...", 
                                key=f"input_{item['id']}",
                                label_visibility="collapsed"
                            )
                            if new_val:
                                items_to_modify.append({'bbox': item['bbox'], 'new_text': new_val})
            elif uploaded:
                st.info("Click 'Scan Text' to detect price tags.")
            
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            if st.button("✨ Render Image", type="primary", use_container_width=True):
                if items_to_modify:
                    with st.spinner("Processing..."):
                        # Re-pass file_bytes to ensure fresh read
                        res = VisionEngine.replace_text_items(file_bytes, items_to_modify, blur_v)
                        st.session_state['final_result'] = res
                else:
                    st.toast("No text changes entered.")

        # --- RIGHT PANEL: VIEW ---
        with c_right:
            st.markdown("#### Preview")
            if 'final_result' in st.session_state:
                st.image(st.session_state['final_result'], caption="Result", use_column_width=True)
                
                # Download
                buf = io.BytesIO()
                st.session_state['final_result'].save(buf, format="PNG")
                st.download_button("Download High-Res", buf.getvalue(), "price_craft.png", "image/png", type="primary", use_container_width=True)
            else:
                st.image(img_fix, caption="Original", use_column_width=True)

if __name__ == "__main__":
    main()
