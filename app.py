import streamlit as st
from streamlit_folium import st_folium
import folium
import os
import cv2
import numpy as np
import torch
from PIL import Image
import mercantile
from simple_satellite_demo import fetch_satellite_tile, fetch_osm_line_pixels, HYPERPARAMETERS, parse_dms
from simple_model_defs import MaskedUNet

import requests
import datetime

# ... (imports)

@st.cache_data
def get_wayback_versions():
    """Fetch available Esri Wayback versions."""
    try:
        url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer?f=json"
        data = requests.get(url).json()
        # Parse 'Selection' list
        # Format: {'Name': 'World Imagery (Wayback 2023-02-08)', 'M': '12345', ...}
        versions = []
        for item in data['Selection']:
            # Extract date from name "World Imagery (Wayback YYYY-MM-DD)"
            name = item['Name']
            if "Wayback" in name:
                date_str = name.split("Wayback ")[-1].replace(")", "")
                versions.append({
                    "date": date_str,
                    "id": item['M'],
                    "label": f"{date_str}"
                })
        return versions
    except Exception as e:
        st.error(f"Failed to fetch Wayback versions: {e}")
        return []

# Set page config
st.set_page_config(page_title="Road Damage Assessment", layout="wide")

st.title("🛰️ Satellite Road Damage Assessment")
st.markdown("Select a location and **date** to analyze road conditions using the RDA UNet model.")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    # Wayback Selection
    wayback_versions = get_wayback_versions()
    selected_wayback = None
    
    if wayback_versions:
        # Default to latest or specific date if requested?
        # Let's verify if user wants 2023-02-07
        
        # Sort by date descending
        wayback_versions.sort(key=lambda x: x['date'], reverse=True)
        
        options = [v['label'] for v in wayback_versions]
        selected_option = st.selectbox("Satellite Imagery Date", options)
        
        # Find selected item
        selected_wayback = next((v for v in wayback_versions if v['label'] == selected_option), None)
        
        if selected_wayback:
            st.info(f"Selected Archive ID: {selected_wayback['id']}")
            
            # Helper to check if it's close to 2023 Earthquake
            if "2023-02" in selected_wayback['date']:
                st.warning("⚠️ You selected a date near the 2023 Earthquake. Note that satellite coverage varies by location and cloud cover.")
    else:
        st.warning("Could not load historical dates. Using latest imagery.")

    # Model Path
    default_model = "models/CRASAR_CRASAR-U-DROIDs-RDA_AAAI26_Simple_VanillaUNet/RDA_UNet_simple_noattention-epoch=05-step=3000-val_macro_iou=0.32182.ckpt"
    model_path = st.text_input("Model Checkpoint Path", value=default_model)
    
    zoom_level = st.slider("Zoom Level", min_value=15, max_value=20, value=18, help="If you see 'Map Data Not Available', try reducing the zoom level.")
    
    analyze_btn = st.button("Analyze Selected Location", type="primary")

# Main Layout
col1, col2 = st.columns([1, 1])

# Initial Map Center (Antakya)
DEFAULT_LAT = 36.20
DEFAULT_LON = 36.16

with col1:
    st.subheader("1. Select Location")
    
    # Configure Map with Wayback Layer if selected
    m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=13)
    
    if selected_wayback:
        # Add Wayback Tile Layer
        # URL: https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{M}/{z}/{y}/{x}
        tile_url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{selected_wayback['id']}/{{z}}/{{y}}/{{x}}"
        folium.TileLayer(
            tiles=tile_url,
            attr=f"Esri Wayback {selected_wayback['date']}",
            name=f"Wayback {selected_wayback['date']}",
            overlay=False
        ).add_to(m)
    else:
        # Default Esri/OpenStreetMap
        pass # Folium defaults to OSM, or we can add Esri World Imagery default
        
    # Add click marker support
    m.add_child(folium.LatLngPopup())
    
    output = st_folium(m, height=500, width="100%")

    selected_lat = None
    selected_lon = None

    if output["last_clicked"]:
        selected_lat = output["last_clicked"]["lat"]
        selected_lon = output["last_clicked"]["lng"]
        st.success(f"Selected: {selected_lat:.6f}, {selected_lon:.6f}")
    else:
        st.info("Click on the map to select a coordinate.")

with col2:
    st.subheader("2. Analysis Result")
    
    if analyze_btn and selected_lat and selected_lon:
        with st.spinner(f"Fetching satellite data ({selected_wayback['date'] if selected_wayback else 'Latest'}) and running analysis..."):
            try:
                # 1. Fetch
                wayback_id = selected_wayback['id'] if selected_wayback else None
                img, bounds = fetch_satellite_tile(selected_lat, selected_lon, zoom_level, wayback_id=wayback_id)
                
                if img is None:
                    st.error("Failed to fetch satellite imagery.")
                else:
                    # Show Original
                    st.image(img, caption="Satellite Input", use_container_width=True)
                    
                    # 2. Process
                    w, h = img.size
                    road_polygons = fetch_osm_line_pixels(bounds, w, h)
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for poly in road_polygons:
                        cv2.polylines(mask, [poly], False, 1, thickness=40)
                        
                    img_np = np.array(img)
                    img_float = img_np.astype(np.float32) / 255.0
                    mask_float = mask.astype(np.float32)
                    
                    input_data = np.dstack([img_float, mask_float])
                    input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0)
                    
                    # 3. Load Model
                    model = MaskedUNet(n_channels=4, n_classes=4, hyperparameters=HYPERPARAMETERS, 
                                    input_channel_mask_index=3, output_channel_background_index=0)
                    
                    checkpoint = torch.load(model_path, map_location="cpu")
                    model.load_state_dict_from_ckpt(checkpoint["state_dict"])
                    model.eval()
                    
                    # 4. Inference
                    with torch.no_grad():
                        preds = model(input_tensor)
                        pred_labels = torch.argmax(preds, dim=1).squeeze().numpy()
                        
                    # 5. Visualize
                    vis_img = img_np.copy()
                    
                    # Colors
                    colors = {
                        1: [0, 255, 0],    # Road (Green)
                        2: [255, 255, 0],  # Partial (Yellow)
                        3: [255, 0, 0]     # Total (Red)
                    }
                    
                    overlay = vis_img.copy()
                    combined_mask = np.zeros(pred_labels.shape, dtype=bool)
                    
                    for label, color in colors.items():
                        mask_layer = (pred_labels == label)
                        overlay[mask_layer] = color
                        combined_mask |= mask_layer
                        
                    alpha = 0.5
                    if combined_mask.any():
                        blended = cv2.addWeighted(vis_img, 1-alpha, overlay, alpha, 0)
                        vis_img[combined_mask] = blended[combined_mask]
                        
                    st.image(vis_img, caption="Damage Analysis Overlay", use_container_width=True)
                    
                    # Legend
                    st.markdown("""
                    **Legend:**
                    - <span style='color:green'>■</span> **Green**: Intact Road
                    - <span style='color:yellow'>■</span> **Yellow**: Partial Damage / Obstruction
                    - <span style='color:red'>■</span> **Red**: Total Destruction / Flooding
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                import traceback
                st.text(traceback.format_exc())
    elif analyze_btn:
        st.warning("Please select a location on the map first.")
