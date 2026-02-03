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
    def scan_all_text(image):
        """Scans all text in the image using EasyOCR."""
        try:
            reader = VisionEngine.get_reader()
            img_np = np.array(image.convert('RGB'))
            
            # Run OCR
            # detail=1 returns (bbox, text, prob)
            results = reader.readtext(img_np)
            
            # Format results
            detected_items = []
            for (bbox, text, prob) in results:
                # bbox is list of 4 points [[x,y], [x,y], [x,y], [x,y]]
                # Filter low confidence or empty
                if prob > 0.3 and text.strip():
                    detected_items.append({
                        'id': f"{text}_{bbox[0][0]}", # Unique ID based on text+pos
                        'text': text,
                        'bbox': bbox,
                        'prob': prob
                    })
            return detected_items
        except Exception as e:
            st.error(f"OCR Error: {e}")
            return []

    @staticmethod
    def replace_text_items(image, items_to_modify, blur_rad=0.0):
        """
        Replaces specific text items in the image.
        items_to_modify: list of dict {'bbox': ..., 'new_text': ...}
        """
        img_np = np.array(image.convert('RGB'))
        
        # 1. Inpaint (Remove old text)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        
        for item in items_to_modify:
            box = np.array(item['bbox'], dtype=np.int32)
            cv2.fillConvexPoly(mask, box, 255)
            
        # Dilate mask slightly to cover edges
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        clean_bg = cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA)
        
        # 2. Draw New Text
        pil_img = Image.fromarray(clean_bg)
        txt_layer = Image.new('RGBA', pil_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(txt_layer)
        font_bytes = VisionEngine.get_font()
        
        for item in items_to_modify:
            new_text = item['new_text']
            if not new_text: continue
            
            box = item['bbox'] # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
            # Calculate Height/Width from bbox
            tl, tr, br, bl = box
            height = int(abs(bl[1] - tl[1]))
            # width = int(abs(tr[0] - tl[0]))
            
            # Load Font (Match Height)
            try:
                # Heuristic: Font size is approx 80% of box height
                f_size = int(height * 0.85)
                if f_size < 10: f_size = 10
                font = ImageFont.truetype(font_bytes, f_size) if font_bytes else ImageFont.load_default()
            except: font = ImageFont.load_default()
            
            # Center text in bbox
            text_bbox = draw.textbbox((0, 0), new_text, font=font)
            t_w = text_bbox[2] - text_bbox[0]
            t_h = text_bbox[3] - text_bbox[1]
            
            # Center X, Center Y of the original box
            center_x = (tl[0] + br[0]) / 2
            center_y = (tl[1] + br[1]) / 2
            
            pos_x = center_x - (t_w / 2)
            pos_y = center_y - (t_h / 2) - (text_bbox[1] * 0.1) # Baseline correction
            
            # Draw (Black, high opacity)
            draw.text((pos_x, pos_y), new_text, font=font, fill=(10, 10, 10, 240))

        # 3. Blur Match
        if blur_rad > 0:
            txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(blur_rad))
            
        pil_img.paste(txt_layer, (0,0), txt_layer)
        return pil_img

# --- UI ---

def inject_premium_css():
    st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; color: #111; font-family: -apple-system, sans-serif; }
        
        /* Inputs */
        .stTextInput input {
            background-color: #FFF !important;
            border: 1px solid #E0E0E0 !important;
            color: #111 !important;
            border-radius: 6px !important;
            padding: 8px !important;
        }
        
        /* Buttons */
        button[kind="primary"] {
            background-color: #000 !important; color: white !important; border: none !important;
        }
        
        /* Text Item Row */
        .text-row {
            background: white;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #EEE;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        /* Hide unwanted elements */
        header, footer { visibility: hidden; }
        .block-container { padding-top: 2rem !important; max-width: 1200px !important; }
        
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    inject_premium_css()
    
    # --- AUTH ---
    if 'auth' not in st.session_state: st.session_state.auth = False
    if not st.session_state.auth:
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.title("Price Craft Studio")
            with st.form("login"):
                if st.text_input("Key", type="password") == st.secrets.get("PASSWORD", "apple"):
                    if st.form_submit_button("Login", type="primary"):
                        st.session_state.auth = True
                        st.rerun()
        return

    # --- STATE MANAGEMENT ---
    if 'ocr_results' not in st.session_state: st.session_state.ocr_results = []
    
    # --- UI LAYOUT ---
    st.markdown("### Price Craft <span style='color:#888'>Text Replacement Studio</span>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Image", type=['png','jpg','jpeg'], label_visibility="collapsed")
    
    if uploaded:
        # Load logic
        img_raw = Image.open(uploaded)
        # Handle transparency
        if img_raw.mode in ('RGBA', 'LA') or (img_raw.mode == 'P' and 'transparency' in img_raw.info):
            bg = Image.new("RGB", img_raw.size, (255, 255, 255))
            bg.paste(img_raw, mask=img_raw.convert('RGBA').split()[3])
            img_fix = bg
        else:
            img_fix = img_raw.convert("RGB")
        img_fix = ImageOps.exif_transpose(img_fix)

        c_left, c_right = st.columns([0.4, 0.6])
        
        # --- LEFT: CONTROLS & DATA ---
        with c_left:
            if st.button("🔍 Scan All Text", type="primary", use_container_width=True):
                with st.spinner("Analyzing image structure (OCR)..."):
                    st.session_state.ocr_results = VisionEngine.scan_all_text(img_fix)
            
            # BLUR SETTING
            blur_v = st.slider("Blur Matching (px)", 0.0, 5.0, 0.5, 0.1)

            st.markdown("---")
            st.caption("DETECTED TEXT")
            
            # DYNAMIC FORM
            items_to_modify = []
            
            if st.session_state.ocr_results:
                # Scrollable container for many items
                with st.container(height=500):
                    for i, item in enumerate(st.session_state.ocr_results):
                        # Layout: Detected Text | Arrow | Input for New
                        c1, c2, c3 = st.columns([0.3, 0.1, 0.6])
                        with c1:
                            st.text_input(f"Org-{i}", value=item['text'], disabled=True, label_visibility="collapsed")
                        with c2:
                            st.markdown("➔")
                        with c3:
                            new_val = st.text_input(
                                f"New-{i}", 
                                placeholder="Keep original", 
                                key=f"input_{item['id']}",
                                label_visibility="collapsed"
                            )
                            if new_val and new_val != "":
                                items_to_modify.append({
                                    'bbox': item['bbox'],
                                    'new_text': new_val
                                })
            else:
                st.info("Click 'Scan All Text' to begin.")
                
            st.markdown("---")
            if st.button("✨ Apply Changes", type="primary", use_container_width=True):
                if items_to_modify:
                    with st.spinner("Rendering changes..."):
                        res = VisionEngine.replace_text_items(img_fix, items_to_modify, blur_v)
                        st.session_state['final_result'] = res
                else:
                    st.toast("No changes entered.")

        # --- RIGHT: PREVIEW ---
        with c_right:
            tab_view = st.tabs(["Result", "Original"])
            with tab_view[0]:
                if 'final_result' in st.session_state:
                    st.image(st.session_state['final_result'], caption="Modified Image", use_column_width=True)
                    
                    # Download
                    buf = io.BytesIO()
                    st.session_state['final_result'].save(buf, format="PNG")
                    st.download_button("Download Image", buf.getvalue(), "modified_price.png", "image/png", type="primary")
                else:
                    st.image(img_fix, caption="Original Image", use_column_width=True)
            
            with tab_view[1]:
                st.image(img_fix, use_column_width=True)

if __name__ == "__main__":
    main()
