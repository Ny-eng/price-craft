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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from streamlit_drawable_canvas import st_canvas
import easyocr
import google.generativeai as genai

# --- PHASE 1: Security & Infrastructure (Zero Proof) ---

# 4. Garbage Collection Force Execution
def secure_cleanup():
    """Forces garbage collection and clears large objects."""
    if 'processed_image' in st.session_state:
        del st.session_state['processed_image']
    gc.collect()

# 2. Volatile Memory Architecture: All images handled in RAM (io.BytesIO)
# (Implicit in code usage of Image.open() and io.BytesIO)

class PriceCraftOS:
    """Core System Logic"""
    
    @staticmethod
    def get_font_bytes(font_url="https://github.com/googlefonts/noto-cjk/raw/main/Sans/Variable/ttf/NotoSansCJKjp-VF.ttf"):
        """Downloads font to memory (Requirements 2 & 42)"""
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

# --- PHASE 2: Apple HIG UI/UX (Refined High Contrast) ---

def inject_apple_css():
    st.markdown("""
    <style>
        /* Force LIGHT Mode & High Contrast */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
        
        :root {
            --bg-color: #F5F5F7;
            --card-bg: #FFFFFF;
            --text-primary: #1D1D1F;
            --text-secondary: #86868B;
            --accent-blue: #0071E3;
            --border-color: #D2D2D7;
        }

        /* Force entire app to light mode colors regardless of user settings */
        .stApp {
            background-color: var(--bg-color) !important;
            color: var(--text-primary) !important;
        }

        /* Streamlit overrides for inputs to be strictly clear */
        .stTextInput input, .stSelectbox div[data-baseweb="select"], .stNumberInput input {
            background-color: #FFFFFF !important;
            color: #1D1D1F !important;
            border: 1px solid #D2D2D7 !important;
            border-radius: 12px !important;
            caret-color: #0071E3 !important;
        }
        
        /* Selectbox dropdown items */
        ul[data-baseweb="menu"] {
            background-color: #FFFFFF !important;
        }
        li[data-baseweb="option"] {
            color: #1D1D1F !important;
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #1D1D1F !important;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        p, label, span, div {
            color: #1D1D1F !important; /* Ensure visibility */
        }
        .caption {
            color: #86868B !important;
        }

        /* Cards */
        .apple-card {
            background-color: #FFFFFF !important; /* Solid white for contrast */
            border-radius: 18px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Stronger shadow */
            padding: 24px;
            border: 1px solid rgba(0,0,0,0.05);
            margin-bottom: 24px;
        }

        /* Buttons */
        .stButton button {
            border-radius: 98px !important;
            font-weight: 500 !important;
            border: none !important;
        }
        .stButton button[kind="primary"] {
            background-color: #0071E3 !important;
            color: #FFFFFF !important;
        }
        .stButton button[kind="secondary"] {
            background-color: #E8E8ED !important;
            color: #1D1D1F !important;
        }

        /* Hide Streamlit Decoration */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem !important;
            max-width: 1280px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- PHASE 3 & 4: Logic & Image Processing ---

def process_image_pipeline(image_bytes, roi_coords, target_price, current_price_text, original_size):
    """
    Executes the 1-pixel compliant processing.
    roi_coords: list of dicts [{'x':, 'y':}, ...] for 4 points.
    """
    
    # 1. Load Image
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = rgb_img.shape[:2]
    
    # Scale coords if canvas was resized
    # Assumes passed coords are normalized or matching the image provided
    # For simplicity in this mono-script, we assume the user selected on the resized view
    # and we pass the scale factor.
    
    # Sort points to logical order: TL, TR, BR, BL
    pts = np.array([[p['x'], p['y']] for p in roi_coords], dtype='float32')
    
    # 22. Mathematical Perspective Absorption
    # Sort points
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    
    # Compute width/height of new rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(rgb_img, M, (maxWidth, maxHeight))
    
    # 23. Texture Completion (Inpainting)
    # Create mask for the text (simple assumption: center area or based on OCR boxes if we had them)
    # For manual, we can just inpaint the whole warped area cautiously?
    # No, that erases background. Better to inpaint specific color range or just the center.
    # Approach: Inpaint the whole "price text" area. We assume the warped box IS the text box.
    mask = np.zeros(warped.shape[:2], dtype=np.uint8)
    # Leave a small border
    pad = 5
    cv2.rectangle(mask, (pad, pad), (maxWidth-pad, maxHeight-pad), 255, -1)
    
    inpainted = cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA) # or NS
    
    # 24/25. Text Rendering with PIL
    pil_img = Image.fromarray(inpainted)
    draw = ImageDraw.Draw(pil_img)
    
    # Font Logic
    font_bytes = PriceCraftOS.get_font_bytes()
    try:
        font_size = int(maxHeight * 0.8) # Heuristic
        if font_bytes:
            font = ImageFont.truetype(font_bytes, font_size)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
        
    # Text Placement
    text = target_price
    # Simple Centering
    # Note: PIL textsize is deprecated, use standard bbox
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    x = (maxWidth - text_w) / 2
    y = (maxHeight - text_h) / 2
    
    # Color logic (Hardcoded black/red for now, or Gemini determined)
    # Assuming 'black' default or extracting from original center
    # 19. Specular/Reflection omitted for brevity in this step, but inpainting preserves some background noise
    draw.text((x, y - top*0.2), text, font=font, fill=(50, 50, 50, 230)) # Slightly transparent for blending
    
    # Warp back
    res_warped_back = cv2.warpPerspective(np.array(pil_img), M, (w, h), flags=cv2.WARP_INVERSE_MAP)
    
    # 26. 1 Pixel Invariant mask
    # We only want to replace the pixels inside the quad.
    mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, rect.astype(int), 255)
    
    # 27. Feathering
    # Blur mask for feathering
    mask_blur = cv2.GaussianBlur(mask_full, (3, 3), 0)
    
    # Composite
    # Convert back to float for blending
    img_float = rgb_img.astype(float)
    res_float = res_warped_back.astype(float)
    mask_float = mask_blur.astype(float) / 255.0
    mask_float = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
    
    final = (res_float * mask_float) + (img_float * (1.0 - mask_float))
    final = final.astype(np.uint8)
    
    return Image.fromarray(final)


# --- MAIN APP ---

def main():
    st.set_page_config(
        page_title="Price Craft",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    inject_apple_css()

    # 3. Password Gateway
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    SYSTEM_PASSWORD = st.secrets.get("PASSWORD", "apple")

    if not st.session_state.authenticated:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center;'>
                <h1>Price Craft</h1>
                <p style='color: #86868B !important;'>National-Grade Precision Engine</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="apple-card">', unsafe_allow_html=True)
                pwd = st.text_input("Enter Access Key", type="password", key="auth_pwd")
                if st.button("Unlock System", type="primary", use_container_width=True):
                    if pwd == SYSTEM_PASSWORD:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Access Denied")
                st.markdown('</div>', unsafe_allow_html=True)
        return

    # --- Authenticated View ---
    
    # Hero Header
    st.markdown("""
    <div style='text-align: center; margin-bottom: 40px;'>
        <h1 style='font-size: 40px;'>Price Craft</h1>
        <p style='font-size: 18px; color: #86868B !important;'>Precision Image Synthesis & Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if 'onboarded' not in st.session_state:
        st.session_state.onboarded = True

    # 1. Upload Section
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if not uploaded_file:
         st.markdown("""
         <div style='text-align: center; padding: 40px; border: 2px dashed #D2D2D7; border-radius: 12px; color: #86868B;'>
            Drag & Drop Price Tag Image Here
         </div>
         """, unsafe_allow_html=True)

    if uploaded_file:
        main_col, control_col = st.columns([0.65, 0.35], gap="large")
        
        # Ensure RGBA/RGB consistency
        image = Image.open(uploaded_file).convert("RGB")
        w, h = image.size
        
        # --- LEFT: Workspace ---
        with main_col:
            st.markdown("### Workspace")
            
            # Logic to fit image nicely
            max_canvas_width = 800
            scale = 1.0
            if w > max_canvas_width:
                scale = max_canvas_width / w
                new_w = max_canvas_width
                new_h = int(h * scale)
            else:
                new_w, new_h = w, h
            
            # Canvas - Force explicit background
            canvas_result = st_canvas(
                fill_color="rgba(0, 113, 227, 0.2)",
                stroke_width=2,
                stroke_color="#0071E3",
                background_image=image, # Now guaranteed RGB
                update_streamlit=True,
                height=new_h,
                width=new_w,
                drawing_mode="polygon",
                key="canvas",
                display_toolbar=True
            )
            st.caption("Draw strictly around the price area using the Polygon tool.")

            # Result View
            if 'processed_image' in st.session_state:
                st.markdown("---")
                st.markdown("### Crafted Result")
                from streamlit_image_comparison import image_comparison
                image_comparison(
                    img1=image,
                    img2=st.session_state['processed_image'],
                    label1="Original",
                    label2="Crafted",
                    width=new_w,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )

        # --- RIGHT: Inspector ---
        with control_col:
            st.markdown('<div class="apple-card">', unsafe_allow_html=True)
            st.markdown("### Configuration")
            
            base_price = st.text_input("New Price", "298", help="Numeric value only")
            
            c_tax, c_lens = st.columns(2)
            with c_tax:
                tax_option = st.selectbox("Tax", ["None", "8%", "10%"])
            with c_lens:
                st.write("") 
                st.write("")
                lens_corr = st.checkbox("Lens Fix")

            # Logic check
            final_display_text = base_price
            if base_price.isdigit() and tax_option != "None":
                rate = 1.08 if tax_option == "8%" else 1.10
                tax_inc = int(int(base_price) * rate)
                final_display_text = f"¥{base_price} (税込¥{tax_inc})"
            else:
                final_display_text = f"¥{base_price}"
            
            st.markdown(f"""
            <div style='background: #F5F5F7; padding: 12px; border-radius: 8px; text-align: center; margin: 15px 0; border: 1px solid #E5E5E5;'>
                <span style='color: #86868B; font-size: 12px;'>PREVIEW RENDER</span><br>
                <strong style='font-size: 20px; color: #1D1D1F;'>{final_display_text}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis
            if st.button("Analyze with Gemini", use_container_width=True):
                 try:
                    with st.spinner("Connecting to Neural Engine..."):
                        # Gemini Fallback Logic
                        api_key = st.secrets.get("GEMINI_API_KEY")
                        if api_key:
                            genai.configure(api_key=api_key)
                            
                            # Try 1.5 Flash first, fallback to Pro Vision
                            model_name = 'gemini-1.5-flash-latest'
                            try:
                                model = genai.GenerativeModel(model_name)
                                # Test generic call or just proceed
                            except:
                                model_name = 'gemini-1.5-flash' # try without latest
                            
                            buf = io.BytesIO()
                            image.save(buf, format="JPEG")
                            prompt = "Analyze font style, weight, and context of this price tag. Return pure JSON."
                            
                            try:
                                response = model.generate_content([{'mime_type': 'image/jpeg', 'data': buf.getvalue()}, prompt])
                                st.success("Analysis Successful")
                                st.json(response.text)
                            except Exception as g_err:
                                # Fallback to Pro Vision if 404/Not Found occurs
                                try:
                                    st.warning(f"Retrying with Legacy Model... ({g_err})")
                                    model = genai.GenerativeModel('gemini-pro-vision')
                                    response = model.generate_content([{'mime_type': 'image/jpeg', 'data': buf.getvalue()}, prompt])
                                    st.success("Analysis Successful (Legacy)")
                                    st.json(response.text)
                                except Exception as e2:
                                    st.error(f"AI Analysis Failed: {e2}")
                        else:
                            st.warning("No API Key Configured")
                 except Exception as e:
                     st.error(f"System Error: {e}")

            st.markdown("---")

            # Main Action
            if st.button("Generate Image", type="primary", use_container_width=True):
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                     obj = canvas_result.json_data["objects"][0]
                     path = obj["path"]
                     points = []
                     for p in path:
                        if p[0] == 'M' or p[0] == 'L':
                            points.append({'x': p[1] / scale, 'y': p[2] / scale})
                     
                     if len(points) >= 3:
                        with st.spinner("Rendering Pixels..."):
                            processed = process_image_pipeline(uploaded_file, points[:4], final_display_text, "0", (w, h))
                            st.session_state['processed_image'] = processed
                            st.rerun()
                else:
                    st.error("Select Area First")

            # Download
            if 'processed_image' in st.session_state:
                 buf = io.BytesIO()
                 exif_data = image.info.get('exif')
                 if exif_data:
                     st.session_state['processed_image'].save(buf, format="PNG", exif=exif_data)
                 else:
                     st.session_state['processed_image'].save(buf, format="PNG")
                 
                 st.markdown("<br>", unsafe_allow_html=True)
                 st.download_button("Download Result", buf.getvalue(), "PriceCraft_Export.png", "image/png", use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

    PriceCraftOS.cleanup()

if __name__ == "__main__":
    main()
