import streamlit as st
import cv2
import numpy as np
import io
import math
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from streamlit_drawable_canvas import st_canvas
import google.generativeai as genai

# --- CONSTANTS & CONFIG ---
PAGE_TITLE = "Price Craft"
PAGE_ICON = "🏷️"

# --- CORE LOGIC ---

class VisionEngine:
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
    def calculate_tilt(pts):
        """Calculate the average tilt angle of the polygon."""
        # pts is list of dicts {'x':, 'y':}
        # Assuming sorted TL, TR, BR, BL order generally, but we just need horizontal text lines.
        # Let's take the first two points as Top Edge if sorted.
        p = sorted(pts, key=lambda z: z['y']) # Sort by Y to find top 2
        top_two = sorted(p[:2], key=lambda z: z['x']) # Sort those by X
        
        dx = top_two[1]['x'] - top_two[0]['x']
        dy = top_two[1]['y'] - top_two[0]['y']
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    @staticmethod
    def auto_detect(image):
        """Robust countour detection."""
        try:
            img = np.array(image.convert('RGB'))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blur, 50, 200)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4 and cv2.contourArea(c) > 2000:
                    points = [{'x': int(p[0][0]), 'y': int(p[0][1])} for p in approx]
                    angle = VisionEngine.calculate_tilt(points)
                    return points, angle
            return None, 0.0
        except: return None, 0.0

    @staticmethod
    def render(img_file, roi, text, blur_rad, original_size):
        """Pipeline: Load -> Warp -> Inpaint -> Text -> Unwarp -> Blend."""
        # Load
        nparr = np.frombuffer(img_file.getvalue(), np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Warp
        pts = np.array([[p['x'], p['y']] for p in roi], dtype='float32')
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        
        mW = int(max(np.linalg.norm(br-bl), np.linalg.norm(tr-tl)))
        mH = int(max(np.linalg.norm(tr-br), np.linalg.norm(tl-bl)))
        dst = np.array([[0, 0], [mW-1, 0], [mW-1, mH-1], [0, mH-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_rgb, M, (mW, mH))
        
        # Inpaint
        mask = np.zeros(warped.shape[:2], dtype=np.uint8)
        pad = 6
        cv2.rectangle(mask, (pad, pad), (mW-pad, mH-pad), 255, -1)
        clean = cv2.inpaint(warped, mask, 3, cv2.INPAINT_TELEA)
        
        # Text
        pil_c = Image.fromarray(clean)
        txt_l = Image.new('RGBA', pil_c.size, (255, 255, 255, 0))
        d = ImageDraw.Draw(txt_l)
        font_b = VisionEngine.get_font()
        try:
            # Dynamic Font Sizing
            f_size = int(mH * 0.85)
            font = ImageFont.truetype(font_b, f_size) if font_b else ImageFont.load_default()
        except: font = ImageFont.load_default()
        
        bbox = d.textbbox((0, 0), text, font=font)
        # Center text
        d.text(( (mW-(bbox[2]-bbox[0]))/2, (mH-(bbox[3]-bbox[1]))/2 - bbox[1]*0.1 ), text, font=font, fill=(10, 10, 10, 240))
        
        if blur_rad > 0:
            txt_l = txt_l.filter(ImageFilter.GaussianBlur(blur_rad))
            
        pil_c.paste(txt_l, (0,0), txt_l)
        
        # Unwarp
        res_warp = cv2.warpPerspective(np.array(pil_c), M, (img_rgb.shape[1], img_rgb.shape[0]), flags=cv2.WARP_INVERSE_MAP)
        
        # Blend
        full_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(full_mask, rect.astype(int), 255)
        # Soften edges
        full_mask = cv2.GaussianBlur(full_mask, (3, 3), 0)
        
        img_f = img_rgb.astype(float)
        res_f = res_warp.astype(float)
        alpha = np.repeat((full_mask.astype(float)/255.0)[:, :, np.newaxis], 3, axis=2)
        
        final = (res_f * alpha) + (img_f * (1.0 - alpha))
        return Image.fromarray(final.astype(np.uint8))

# --- UI ---

def inject_premium_css():
    st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; color: #111; font-family: -apple-system, sans-serif; }
        
        /* Containers */
        .main-card {
            background: white; padding: 2rem; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1rem;
        }
        
        /* Text Inputs */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #FFF !important;
            border: 1px solid #E0E0E0 !important;
            color: #111 !important;
            border-radius: 6px !important;
            padding: 10px !important;
        }
        
        /* Primary Button */
        div[data-testid="stButton"] button {
            border: 1px solid #E0E0E0; font-weight: 600; color: #333;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #000; color: #000; background: #FFF;
        }
        
        /* Accent Button (Primary) */
        button[kind="primary"] {
            background-color: #000 !important; color: white !important; border: none !important;
        }
        button[kind="primary"]:hover {
            background-color: #333 !important;
        }
        
        /* Metrics */
        .metric-box {
            background: #F1F3F5; padding: 0.8rem; border-radius: 6px; 
            text-align:center; font-size: 14px; font-weight: 500; color: #444;
        }
        .metric-val { font-size: 18px; font-weight: 700; color: #000; }
        
        /* Hide */
        [data-testid="stSidebar"] { display: none; } /* NO SIDEBAR - Full Width Design */
        header, footer { visibility: hidden; }
        .block-container { padding-top: 2rem !important; max-width: 1000px !important; }
        
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")
    inject_premium_css()
    
    # --- AUTH ---
    if 'auth' not in st.session_state: st.session_state.auth = False
    if not st.session_state.auth:
        st.markdown("<h1 style='text-align:center; margin-bottom: 2rem'>Price Craft OS</h1>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            with st.form("login"):
                p = st.text_input("Access Key", type="password")
                if st.form_submit_button("Enter", use_container_width=True, type="primary"):
                    if p == st.secrets.get("PASSWORD", "apple"):
                        st.session_state.auth = True
                        st.rerun()
                    else: st.error("Mismatched Key")
        return

    # --- APP STATE ---
    if 'scan_res' not in st.session_state: st.session_state.scan_res = {}
    
    # --- LAYOUT ---
    st.markdown("## Price Craft <span style='font-size:14px; color:#888; font-weight:400'>Studio</span>", unsafe_allow_html=True)
    
    # 1. UPLOAD AREA
    with st.container():
        uploaded = st.file_uploader("Source Image", type=['png','jpg','jpeg'], label_visibility="collapsed")

    if uploaded:
        file_id = f"{uploaded.file_id}"
        img_raw = Image.open(uploaded).convert("RGB")
        img_fix = ImageOps.exif_transpose(img_raw)
        
        # 2. WORKSPACE
        c_left, c_right = st.columns([0.35, 0.65])
        
        # --- LEFT PANEL: CONTROLS ---
        with c_left:
            st.markdown("### Controls")
            
            # SCAN BUTTON
            if st.button("✨ Auto-Analyze", type="primary", use_container_width=True):
                with st.spinner("Vision Engine Running..."):
                    # 1. Geometry
                    poly, angle = VisionEngine.auto_detect(img_fix)
                    st.session_state.scan_res['poly'] = poly
                    st.session_state.scan_res['angle'] = angle
                    
                    # 2. Text (Mock AI)
                    st.session_state.scan_res['text'] = "298" 
                    st.session_state.scan_res['conf'] = "High"
            
            # RESULTS DASHBOARD
            if 'text' in st.session_state.scan_res:
                r = st.session_state.scan_res
                
                # Metrics Row
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"<div class='metric-box'>Detected<br><span class='metric-val'>{r.get('text','--')}</span></div>", unsafe_allow_html=True)
                with m2:
                    ang = r.get('angle', 0.0)
                    st.markdown(f"<div class='metric-box'>Tilt<br><span class='metric-val'>{ang:.1f}°</span></div>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # EDIT FORM
                target_p = st.text_input("New Price", value=r.get('text', '298'))
                tax_opt = st.selectbox("Tax Mode", ["None", "8%", "10%"])
                
                blur_v = st.slider("Blur Intensity", 0.0, 5.0, 0.5, 0.1)
                
                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                if st.button("Render Final Asset", type="primary", use_container_width=True):
                    st.session_state['trigger_render'] = True
            
            else:
                st.info("Upload image and click Auto-Analyze to begin.")

        # --- RIGHT PANEL: CANVAS & PREVIEW ---
        with c_right:
            tab1, tab2 = st.tabs(["Editor", "Result"])
            
            with tab1:
                # CANVAS SETUP
                # Responsive width
                disp_w = 600
                scale = disp_w / img_fix.width
                disp_h = int(img_fix.height * scale)
                
                disp_img = img_fix.resize((disp_w, disp_h))
                
                # Initial Drawing from Auto-Detect
                init_draw = None
                if 'poly' in st.session_state.scan_res:
                    pts = st.session_state.scan_res['poly']
                    if pts:
                        # Scale points
                        s_pts = [[p['x']*scale, p['y']*scale] for p in pts]
                        path_svg = [["M", s_pts[0][0], s_pts[0][1]], 
                                    ["L", s_pts[1][0], s_pts[1][1]],
                                    ["L", s_pts[2][0], s_pts[2][1]],
                                    ["L", s_pts[3][0], s_pts[3][1]],
                                    ["Z"]]
                        init_draw = {
                            "version": "4.4.0",
                            "objects": [{
                                "type": "path", "originX": "left", "originY": "top", "left": 0, "top": 0,
                                "fill": "rgba(255, 0, 0, 0.2)", "stroke": "red", "strokeWidth": 2,
                                "path": path_svg
                            }]
                        }
                
                st.caption("Adjust the Red Box to match the price tag.")
                canvas = st_canvas(
                    fill_color="rgba(255, 0, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="red",
                    background_image=disp_img,
                    initial_drawing=init_draw,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="polygon",
                    key=f"editor_{file_id}_{st.session_state.scan_res.get('angle',0)}"
                )
                
            with tab2:
                if st.session_state.get('trigger_render', False):
                    # EXECUTE RENDER
                    valid_roi = None
                    # 1. Try get from Canvas (User Edit)
                    if canvas.json_data and canvas.json_data['objects']:
                         path = canvas.json_data['objects'][0]['path']
                         # Filter 'M' and 'L' commands
                         pts = [p for p in path if p[0] in ['M','L']]
                         if len(pts) >= 4:
                             valid_roi = [{'x': p[1]/scale, 'y': p[2]/scale} for p in pts[:4]]
                    
                    if valid_roi:
                        # Format Text
                        u_price = target_p
                        if u_price.isdigit() and tax_opt != "None":
                            r = 1.08 if tax_opt == "8%" else 1.10
                            u_price = f"¥{u_price} (税込¥{int(int(u_price)*r)})"
                        elif u_price.isdigit():
                            u_price = f"¥{u_price}"
                            
                        final_img = VisionEngine.render(uploaded, valid_roi, u_price, blur_v, img_fix.size)
                        st.session_state['final_out'] = final_img
                    st.session_state['trigger_render'] = False # Reset
                
                if 'final_out' in st.session_state:
                    st.image(st.session_state['final_out'], use_column_width=True)
                    
                    buf = io.BytesIO()
                    st.session_state['final_out'].save(buf, format="PNG")
                    st.download_button("Download High-Res", buf.getvalue(), "price_craft.png", "image/png", type="primary", use_container_width=True)
                else:
                    st.info("Click 'Render Final Asset' to generate.")

if __name__ == "__main__":
    main()
