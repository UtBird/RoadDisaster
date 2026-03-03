import streamlit as st
from streamlit_folium import st_folium
import folium
import os
import cv2
import numpy as np
import torch
from PIL import Image
import mercantile
import requests
import datetime
import traceback

# Hardcoded Hyperparameters for AttentionUNet (Full)
HYPERPARAMETERS = {
    "input": {
        "model_parameters": {
            "layers": {
                "inc": {"out_channels": 32, "dilation": 1, "kernel_size": 5, "padding_mode": "zeros"},
                "down_1": {"in_channels": 32, "out_channels": 64, "dilation": 1, "kernel_size": 5, "padding_mode": "zeros"},
                "down_2": {"in_channels": 64, "out_channels": 128, "dilation": 1, "kernel_size": 5, "padding_mode": "zeros"},
                "down_3": {"in_channels": 128, "out_channels": 256, "dilation": 1, "kernel_size": 5, "padding_mode": "zeros"},
                "down_4": {"in_channels": 256, "out_channels": 512, "dilation": 1, "kernel_size": 5, "padding_mode": "zeros"},
                "down_5": {"in_channels": 512, "out_channels": 1024, "dilation": 1, "kernel_size": 5, "padding_mode": "zeros"},
                "up_0": {"attention": True, "bilinear": False, "in_channels": 1024, "out_channels": 512},
                "up_1": {"attention": True, "bilinear": False, "in_channels": 512, "out_channels": 256},
                "up_2": {"attention": True, "bilinear": False, "in_channels": 256, "out_channels": 128},
                "up_3": {"attention": True, "bilinear": False, "in_channels": 128, "out_channels": 64},
                "up_4": {"attention": True, "bilinear": False, "in_channels": 64, "out_channels": 32}
            }
        }
    }
}
# Import project utilities
from simple_satellite_demo import fetch_satellite_tile, fetch_osm_line_pixels, parse_dms
from src.map_utils import search_oam_images
from simple_model_defs import MaskedUNet

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Road Damage Assessment",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    h1 { color: #2E86C1; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #2E86C1; }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data
def get_wayback_versions():
    """Fetch available Esri Wayback versions."""
    try:
        url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer?f=json"
        data = requests.get(url, timeout=5).json()
        versions = []
        for item in data.get('Selection', []):
            name = item['Name']
            if "Wayback" in name:
                date_str = name.split("Wayback ")[-1].replace(")", "")
                versions.append({
                    "date": date_str,
                    "id": item['M'],
                    "label": f"{date_str}"
                })
        return versions
    except Exception:
        return []

def calculate_auto_threshold(damage_probs, road_mask, bias=0.0):
    """
    Enhanced Optimization: Peak-Aware Thresholding.
    Determines if the road is clean before calculating a split.
    """
    road_pixels = damage_probs[road_mask > 0]
    
    if len(road_pixels) < 50:
        return 0.5 + bias
    
    # 1. Cleanliness Check: We only reject as "clean" if the signal is EXTREMELY flat.
    p95 = np.percentile(road_pixels, 95)
    mean_val = np.mean(road_pixels)
    
    # If the road is virtually silent (95% under 0.05), it's definitely clean.
    # We lowered this from 0.10 to 0.05 to catch very subtle debris.
    if p95 < 0.05:
        return 0.90 
        
    # 2. Otsu calculation (Mathematical split)
    pixels_8bit = (np.clip(road_pixels, 0, 1) * 255).astype(np.uint8)
    otsu_thresh_val, _ = cv2.threshold(pixels_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = float(otsu_thresh_val) / 255.0
    
    # 3. Dynamic Floor based on average noise
    # We want a floor that rejects tiny flickers but keeps real signal.
    # Lowered the baseline floor to 0.08 to be "Ultra-Sensitive".
    # Scaling factor lowered from 1.8x to 1.5x.
    noise_floor = max(0.08, mean_val * 1.5)
    
    # Hybrid Result
    optimized_thresh = max(noise_floor, otsu_thresh)
    
    # Apply user bias (bias is subtracted from threshold to increase sensitivity)
    return np.clip(optimized_thresh + bias, 0.02, 0.95)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛰️ RDA Control Panel")
    st.info("Analyze road damage using AI.")
    
    st.header("1. Location & Data")
    
    # City Selection
    CITIES = {
        "Antakya (Hatay)": [36.20, 36.16],
        "Kahramanmaraş": [37.57, 36.93],
        "Gaziantep": [37.06, 37.38],
        "Adıyaman": [37.76, 38.27],
        "Malatya": [38.35, 38.30],
        "Osmaniye": [37.07, 36.24],
        "Adana": [37.00, 35.32],
        "Şanlıurfa": [37.16, 38.79],
        "Diyarbakır": [37.91, 40.21],
        "Kilis": [36.71, 37.11]
    }
    selected_city_name = st.selectbox("📍 Jump to City", list(CITIES.keys()))
    city_coords = CITIES[selected_city_name]
    
    st.markdown("---")
    
    # Imagery Source
    wayback_versions = get_wayback_versions()
    
    opt_google = "Google Maps (Latest / High Res)"
    opt_oam = "OpenAerialMap (Event Specific)"
    
    wayback_options = []
    wayback_map = {}
    if wayback_versions:
        wayback_versions.sort(key=lambda x: x['date'], reverse=True)
        wayback_map = {f"Esri Wayback ({v['date']})": v for v in wayback_versions}
        wayback_options = list(wayback_map.keys())
    
    source_options = [opt_google, opt_oam] + wayback_options
    selected_source = st.selectbox("📡 Satellite Source", source_options)
    
    # Source State
    provider = "Esri"
    selected_wayback = None
    selected_oam_url = None
    
    if selected_source == opt_google:
        provider = "Google Maps"
    elif selected_source == opt_oam:
        provider = "OpenAerialMap"
        st.caption("Search for post-disaster imagery (e.g. Feb 2023)")
        c1, c2 = st.columns(2)
        d_start = c1.date_input("Start", datetime.date(2023, 2, 6))
        d_end = c2.date_input("End", datetime.date(2023, 2, 28))
        
        if st.button("🔎 Search Images"):
            if 'last_clicked' in st.session_state and st.session_state['last_clicked']:
                lat, lon = st.session_state['last_clicked']
                bbox = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
            else:
                lat, lon = city_coords
                bbox = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
            
            with st.spinner("Searching OAM..."):
                st.session_state['oam_results'] = search_oam_images(bbox, d_start, d_end)
        
        if st.session_state.get('oam_results'):
            res = st.session_state['oam_results']
            st.success(f"Found {len(res)} images.")
            oam_opts = {f"{r['date']} - {r['provider']}": r for r in res}
            sel_oam = st.selectbox("Select Image", list(oam_opts.keys()))
            if sel_oam:
                selected_oam_url = oam_opts[sel_oam]['tms_url']
    else:
        provider = "Esri Wayback"
        selected_wayback = wayback_map.get(selected_source)
    
    st.header("2. Analysis Settings")
    with st.expander("⚙️ Fine-Tuning", expanded=True):
        zoom_level = st.slider("Zoom Level", 15, 20, 18)
        
        st.markdown("#### Analysis Parameters")
        
        damage_booster = st.slider("🔥 Damage Sensitivity Booster", 1.0, 10.0, 3.5, 0.5,
                                 help="Multiplies the AI's confidence for damage categories.")
        
        auto_threshold = st.toggle("🤖 Auto-Optimize Threshold", value=True, help="Uses modern hybrid logic to find the best threshold dynamically.")
        
        if not auto_threshold:
            sensitivity_threshold = st.slider("📉 Detection Threshold", 0.0, 1.0, 0.25, 0.05)
            auto_offset = 0.0
        else:
            sensitivity_threshold = 0.25 # Base
            auto_offset = st.slider("⚖️ Auto-Sensitivity Bias", -0.5, 0.5, 0.0, 0.05, 
                                   help="Adjust the auto-calculated threshold. Plus = More conservative, Minus = More sensitive.")
        
        line_width = st.slider("🖊️ Visualization Width", 5, 40, 15)
        
        st.markdown("#### Model & Stability")
        enable_tta = st.toggle("🚀 Test-Time Augmentation (TTA)", value=True, help="Analyze image at 4 rotations for maximum stability. (Slower but 4x more reliable)")
        enable_smoothing = st.toggle("✨ Smooth Results", value=True, help="Apply Gaussian blur to probabilities to reduce pixel noise.")
        
        default_model = "models/CRASAR_Full_AttentionUNet/RDA_UNet_full_v1-epoch=02-step=1500-val_macro_iou=0.07566.ckpt"
        model_path = st.text_input("📁 Model Checkpoint (.ckpt)", value=default_model)
        
        debug_mode = st.checkbox("Show Debug Logs", value=True)
    
    analyze_btn = st.button("🚀 Start Analysis", type="primary")
    
    if st.button("🔄 Reset Analysis"):
        st.session_state.analysis_result = None
        st.rerun()

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------

st.title("Satellite Road Damage Assessment")

# Map Display
m = folium.Map(location=city_coords, zoom_start=14)

# Add Base Layer
if provider == "Esri Wayback" and selected_wayback:
    url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{selected_wayback['id']}/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(tiles=url, attr=f"Esri {selected_wayback['date']}", name="Wayback").add_to(m)
elif provider == "Google Maps":
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite"
    ).add_to(m)
elif provider == "OpenAerialMap" and selected_oam_url:
    folium.TileLayer(tiles=selected_oam_url, attr="OAM", name="OAM Layer").add_to(m)

m.add_child(folium.LatLngPopup())
output = st_folium(m, height=450, width="100%")

# Coordinate Handling
click_lat, click_lon = None, None
if output.get("last_clicked"):
    click_lat = output["last_clicked"]["lat"]
    click_lon = output["last_clicked"]["lng"]
    st.caption(f"Selected: {click_lat:.5f}, {click_lon:.5f}")
else:
    st.info("👆 Click on the map to select a location.")

st.markdown("---")

# -----------------------------------------------------------------------------
# Trigger Analysis
# -----------------------------------------------------------------------------
if analyze_btn:
    if not click_lat:
        st.warning("Please select a location on the map first.")
    else:
        # Progress UI
        progress_bar = st.progress(0, text="Starting Analysis...")
        status_text = st.empty()
        
        try:
            # 1. Fetch Image
            status_text.text("1/4 Downloading Satellite Imagery...")
            progress_bar.progress(10)
            
            wayback_id = selected_wayback['id'] if selected_wayback else None
            prov_code = 'google' if provider == "Google Maps" else 'custom' if provider == "OpenAerialMap" else 'esri'
            custom_url = selected_oam_url
            
            if debug_mode:
                st.write(f"**DEBUG:** Fetching from Provider: `{prov_code}`")
            
            img, bounds = fetch_satellite_tile(click_lat, click_lon, zoom_level, 
                                             wayback_id=wayback_id, provider=prov_code, custom_url=custom_url)
            
            if debug_mode:
                st.write(f"**DEBUG:** Primary Fetch Status: `{'Success' if img else 'Failed/None'}`")

            # AUTOMATIC FALLBACK LOGIC
            if img is None and prov_code != 'google':
                 status_text.text("Trying Fallback (Google Maps)...")
                 if debug_mode:
                    st.write("**DEBUG:** Attempting Google Maps Fallback...")
                 
                 img, bounds = fetch_satellite_tile(click_lat, click_lon, zoom_level, provider='google')
                 
                 if debug_mode:
                    st.write(f"**DEBUG:** Fallback Fetch Status: `{'Success' if img else 'Failed/None'}`")
                 if img:
                     st.toast("Using Google Maps Fallback", icon="⚠️")
            
            if img is None:
                st.error("FAILED to download imagery.")
                st.warning("👉 Try selecting a different location or check your internet connection.")
                progress_bar.progress(100, text="Aborted.")
                st.stop()
            
            else:
                # 2. OSM Roads
                if debug_mode:
                    st.write("**DEBUG:** Starting OSM Road Fetch...")
                status_text.text("2/4 Fetching Road Network (OSM)...")
                progress_bar.progress(40)
                
                road_polygons = []
                try:
                    w, h = img.size
                    if debug_mode:
                        st.write(f"**DEBUG:** Image Size: {w}x{h}, Bounds: {bounds}")
                    road_polygons = fetch_osm_line_pixels(bounds, w, h)
                    if debug_mode:
                        st.write(f"**DEBUG:** OSM API Result: {len(road_polygons)} road segments found.")
                except Exception as e_osm:
                    st.write(f"**DEBUG:** OSM Fetch CRASHED: {e_osm}")
                    st.code(traceback.format_exc())
                    st.warning("⚠️ Failed to fetch roads. Analysis proceeds without mask.")
                    road_polygons = []

                if not road_polygons:
                     st.warning("⚠️ Could not fetch road data (OSM API might be busy). Analysis will proceed, but road mask will be empty.")
                
                # Create Mask
                if debug_mode:
                    st.write("**DEBUG:** Creating Road Mask...")
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in road_polygons:
                    cv2.polylines(mask, [poly], False, 1, thickness=line_width)
                
                # Prepare Tensors
                status_text.text("3/4 Preprocessing...")
                progress_bar.progress(60)
                
                # Ensure RGB (Removes Alpha channel that could mess up masking)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    
                img_np = np.array(img)
                img_float = img_np.astype(np.float32) / 255.0
                mask_float = mask.astype(np.float32)
                
                input_data = np.dstack([img_float, mask_float])
                
                # 3. Load Model
                status_text.text("Loading AI Model...")
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = MaskedUNet(n_channels=4, n_classes=11, hyperparameters=HYPERPARAMETERS, 
                               input_channel_mask_index=3, output_channel_background_index=0)
            
                    ckpt = torch.load(model_path, map_location=device)
                    # Handle lightning ckpt mapping
                    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
                    model.load_state_dict_from_ckpt(state_dict)
                    model.eval()
                    model.to(device)
                except Exception as e_model:
                     st.error(f"**FATAL ERROR:** Model Load Failed: {e_model}")
                     st.stop()
                
                # 4. Inference
                status_text.text("4/4 Running AI Inference...")
                progress_bar.progress(80)
                
                input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                with torch.no_grad():
                    if enable_tta:
                        # Test-Time Augmentation: Average the probabilities directly
                        # MaskedUNet.forward already handles Softmax and Masking.
                        sum_probs = None
                        for rot in [0, 1, 2, 3]: # 0, 90, 180, 270 degrees
                            aug_input = torch.rot90(input_tensor, k=rot, dims=[2, 3])
                            # Masked probabilities from model
                            pred_probs = model(aug_input) 
                            # Rotate back
                            inv_probs = torch.rot90(pred_probs, k=-rot, dims=[2, 3])
                            
                            if sum_probs is None:
                                sum_probs = inv_probs
                            else:
                                sum_probs += inv_probs
                        
                        probs = (sum_probs / 4.0).squeeze().cpu().numpy()
                    else:
                        # Direct masked probabilities
                        probs = model(input_tensor).squeeze().cpu().numpy()

                    # Class mapping: Background=0, Road=1, Partial=2-6, Total=7-10
                    p_background = probs[0]
                    p_road = probs[1]
                    p_partial = np.sum(probs[2:7], axis=0)
                    p_total = np.sum(probs[7:11], axis=0)
                    
                    # Apply Booster to damage signals
                    p_partial_boosted = np.clip(p_partial * damage_booster, 0, 1)
                    p_total_boosted = np.clip(p_total * damage_booster, 0, 1)

                    if enable_smoothing:
                        p_partial_boosted = cv2.GaussianBlur(p_partial_boosted, (5, 5), 0)
                        p_total_boosted = cv2.GaussianBlur(p_total_boosted, (5, 5), 0)

                    # 1. Initial prediction based on UNBOOSTED natural probabilities
                    # This ensures Background (0) wins if it's the strongest natural signal.
                    natural_scores = np.stack([p_background, p_road, p_partial, p_total])
                    pred_labels = np.argmax(natural_scores, axis=0)
                    
                    # 2. Calculate damage ratio (only for pixels already identified as road/damage)
                    # ratio = (damage) / (total road mass)
                    damage_ratio = (p_partial_boosted + p_total_boosted) / (p_road + p_partial_boosted + p_total_boosted + 1e-6)
                    damage_ratio = np.clip(damage_ratio, 0, 1)
                    
                    # 3. Threshold Optimization
                    actual_threshold = sensitivity_threshold
                    if auto_threshold:
                        actual_threshold = calculate_auto_threshold(damage_ratio, mask, bias=auto_offset)
                        if debug_mode:
                            st.write(f"**DEBUG:** Dynamic Optimization Threshold: `{actual_threshold:.4f}`")

                    # 4. Final Refinement
                    h_res, w_res = pred_labels.shape
                    for r in range(h_res):
                        for c in range(w_res):
                            # IMPORTANT: If model initially said Background (0), WE RESPECT THAT.
                            # We only refine pixels the model thinks are not background.
                            if pred_labels[r, c] != 0: 
                                is_naturally_damaged = pred_labels[r, c] >= 2
                                
                                # Decision: Damage vs Clean Road
                                # We allow a 40% threshold reduction if the model naturally suspected damage
                                effective_thresh = actual_threshold * 0.6 if is_naturally_damaged else actual_threshold
                                
                                if damage_ratio[r, c] > effective_thresh:
                                    # Pick between partial and total
                                    if p_total_boosted[r, c] > p_partial_boosted[r, c]:
                                        pred_labels[r, c] = 3 # Total
                                    else:
                                        pred_labels[r, c] = 2 # Partial
                                else:
                                    pred_labels[r, c] = 1 # Clean Road
                
                # SAVE TO SESSION STATE
                st.session_state.analysis_result = {
                    "original_img": img,
                    "img_np": img_np,
                    "pred_labels": pred_labels,
                    "damage_ratio": damage_ratio,
                    "p_road_max": np.max(p_road),
                    "p_damage_max": np.max(p_total),
                    "p_damage_boosted_max": np.max(p_total_boosted),
                    "booster": damage_booster,
                    "threshold": actual_threshold,
                    "is_auto": auto_threshold
                }
                
                progress_bar.progress(100)
                status_text.text("Analysis Complete!")
                st.rerun() # Refresh to show results
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.code(traceback.format_exc())
            progress_bar.empty()

# -----------------------------------------------------------------------------
# Result Display
# -----------------------------------------------------------------------------
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    
    st.subheader("📝 Analysis Report")
    
    if debug_mode:
        st.success("Analysis persisted successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Satellite**")
        st.image(res["original_img"], width="stretch")
    
    with col2:
        st.markdown("**Damage Detection**")
        vis_img = res["img_np"].copy()
        
        # Colors
        colors = {
            1: [0, 255, 0],    # Green
            2: [255, 215, 0],  # Gold
            3: [220, 20, 60]   # Crimson
        }
        
        overlay = vis_img.copy()
        combined_mask = np.zeros(res["pred_labels"].shape, dtype=bool)
        
        for label, color in colors.items():
            m_layer = (res["pred_labels"] == label)
            overlay[m_layer] = color
            combined_mask |= m_layer
            
        alpha = 0.55
        if combined_mask.any():
            blended = cv2.addWeighted(vis_img, 1-alpha, overlay, alpha, 0)
            vis_img[combined_mask] = blended[combined_mask]
            
        st.image(vis_img, width="stretch")

    # Legend
    st.markdown("""
    <div style="background-color: #0E1117; padding: 15px; border-radius: 8px; border: 1px solid #333; display: flex; justify-content: space-around; margin-top: 10px;">
        <span style="color: #00FF00; font-weight: bold; font-size: 1.1em;">■ Intact Road</span>
        <span style="color: #FFD700; font-weight: bold; font-size: 1.1em;">■ Partial / Blocked</span>
        <span style="color: #DC143C; font-weight: bold; font-size: 1.1em;">■ Destroyed</span>
    </div>
    """, unsafe_allow_html=True)
    
    if debug_mode:
        st.markdown("### 🧩 Diagnostics")
        c1, c2 = st.columns(2)
        
        # Heatmap
        hm = (res["damage_ratio"] * 255).astype(np.uint8)
        hm_c = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_c[res["damage_ratio"] < 0.05] = 0
        c1.image(hm_c, caption="Damage Probability Heatmap", width="stretch")
        
        with c2:
            st.info(f"""
            **Stats:**
            - Booster Factor: `{res['booster']}x`
            - Final Threshold: `{res['threshold']:.3f}` {'(🤖 Hybrid Auto)' if res.get('is_auto') else '(🛠️ Manual)'}
            - Max Road Prob: `{res['p_road_max']:.3f}`
            - Max Damage Ratio: `{np.max(res['damage_ratio']):.3f}`
            """)
            
            # Probability Histogram
            road_data = res["damage_ratio"][res["damage_ratio"] > 0.01]
            if len(road_data) > 0:
                st.caption("Distribution of Suspected Damage (Histogram)")
                hist_vals, bin_edges = np.histogram(road_data, bins=20, range=(0, 1))
                st.bar_chart(hist_vals)
                st.caption(f"Current threshold ({res['threshold']:.2f}) cuts the bars at that level.")
