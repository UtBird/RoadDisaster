import streamlit as st
from streamlit_folium import st_folium
import folium
import os
import cv2
import numpy as np
import torch
import requests
import datetime
import traceback
from PIL import Image
import segmentation_models_pytorch as smp
import math

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
# Helper Functions (Tile Math & Network)
# -----------------------------------------------------------------------------
def num2deg(xtile, ytile, zoom):
    """Google/OSM Tile to Lat/Lon conversion."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_satellite_tile_custom(lat, lon, zoom_level=18, wayback_id=None, provider='google', custom_url=None):
    """Downloads a 512x512 tile around the coordinate by fetching 4 256x256 standard tiles."""
    n = 2.0 ** zoom_level
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    
    stitched_img = Image.new('RGB', (512, 512))
    
    positions = [(0, 0, xtile, ytile), (256, 0, xtile+1, ytile),
                 (0, 256, xtile, ytile+1), (256, 256, xtile+1, ytile+1)]
                 
    for px, py, xt, yt in positions:
        if provider == 'esri' and wayback_id:
            url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{wayback_id}/{zoom_level}/{yt}/{xt}"
        elif provider == 'google':
            url = f"https://mt1.google.com/vt/lyrs=s&x={xt}&y={yt}&z={zoom_level}"
        elif provider == 'custom' and custom_url:
            url = custom_url.replace('{x}', str(xt)).replace('{y}', str(yt)).replace('{z}', str(zoom_level))
        else:
            return None, None
            
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                from io import BytesIO
                tile_img = Image.open(BytesIO(r.content)).convert('RGB')
                stitched_img.paste(tile_img, (px, py))
            else:
                return None, None
        except:
             return None, None

    # Calculate actual bounding box for exactly this 512x512 stitched area
    lat_n, lon_w = num2deg(xtile, ytile, zoom_level)
    lat_s, lon_e = num2deg(xtile + 2, ytile + 2, zoom_level)
    bounds = (lon_w, lat_s, lon_e, lat_n) # min_lon, min_lat, max_lon, max_lat
    return stitched_img, bounds


def get_osm_roads_overpass(bounds, w, h, thickness=4):
    """Fetches OSM roads via Overpass and draws perfectly aligned array."""
    import requests
    west, south, east, north = bounds
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f'''
    [out:json];
    way["highway"]({south},{west},{north},{east});
    out geom;
    '''
    road_img = np.zeros((h, w), dtype=np.uint8)
    try:
        resp = requests.post(overpass_url, data=query, timeout=15)
        data = resp.json()
        for element in data.get('elements', []):
            if 'geometry' in element:
                pts = []
                for pt in element['geometry']:
                    px = int((pt['lon'] - west) / (east - west) * w)
                    py = int((north - pt['lat']) / (north - south) * h)
                    pts.append([px, py])
                if len(pts) >= 2:
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(road_img, [pts], False, 1, thickness=thickness)
    except Exception as e:
        st.warning(f"OSM Fetch error: {e}")
    return road_img

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
                    "date": date_str, "id": item['M'], "label": f"{date_str}"
                })
        return versions
    except Exception:
        return []

@st.cache_resource
def load_simple_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Segformer(encoder_name="mit_b3", encoder_weights=None, in_channels=3, classes=1).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    return None, device

def search_oam_images(bbox, date_start=None, date_end=None, limit=50):
    url = "https://api.openaerialmap.org/meta"
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": limit,
        "order_by": "acquisition_end",
        "sort": "desc"
    }
    
    if date_start and date_end:
        if isinstance(date_start, str): params["acquisition_from"] = date_start
        else: params["acquisition_from"] = date_start.strftime("%Y-%m-%d")
        if isinstance(date_end, str): params["acquisition_to"] = date_end
        else: params["acquisition_to"] = date_end.strftime("%Y-%m-%d")

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        results = []
        if 'results' in data:
            for item in data['results']:
                tms_url = item.get('tms') or item.get('properties', {}).get('tms')
                if not tms_url:
                    tms_url = item.get('wmts') or item.get('properties', {}).get('wmts')
                
                if tms_url:
                    results.append({
                        "id": item.get('_id') or item.get('uuid'),
                        "title": item.get('title', 'Unknown Image'),
                        "provider": item.get('provider', 'Unknown'),
                        "date": item.get('acquisition_end', item.get('acquisition_start', 'Unknown Date')),
                        "tms_url": tms_url,
                        "bbox": item.get('bbox')
                    })
        return results
    except Exception as e:
        st.error(f"OAM Search failed: {e}")
        return []

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛰️ RDA Control Panel")
    st.info("Analyze road damage using AI (Segformer).")
    
    st.header("1. Location & Data")
    
    CITIES = {
        "Antakya (Hatay)": [36.20, 36.16],
        "Kahramanmaraş": [37.57, 36.93],
        "Gaziantep": [37.06, 37.38],
        "Malatya": [38.35, 38.30],
        "Adıyaman": [37.76, 38.27]
    }
    selected_city_name = st.selectbox("📍 Jump to City", list(CITIES.keys()))
    city_coords = CITIES[selected_city_name]
    
    st.markdown("---")
    
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
    
    provider = "Esri Wayback" if "Wayback" in selected_source else "Google Maps"
    selected_wayback = wayback_map.get(selected_source)
    selected_oam_url = None
    
    if selected_source == opt_oam:
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
            res_oam = st.session_state['oam_results']
            st.success(f"Found {len(res_oam)} images.")
            oam_opts = {f"{r['date']} - {r['provider']}": r for r in res_oam}
            sel_oam = st.selectbox("Select Image", list(oam_opts.keys()))
            if sel_oam:
                selected_oam_url = oam_opts[sel_oam]['tms_url']
    
    st.header("2. Analysis Settings")
    with st.expander("⚙️ Fine-Tuning", expanded=True):
        zoom_level = st.slider("Zoom Level", 16, 19, 18)
        
        st.markdown("#### Detection Parameters")
        damage_booster = st.slider("🔥 Damage Sensitivity Booster", 1.0, 10.0, 3.5, 0.5, help="Modelin hasar sinyallerini yapay olarak güçlendirir. Zayıf tahminleri yakalamak için artırın.")
        threshold = st.slider("📉 Detection Threshold", 0.05, 0.95, 0.40, 0.05, help="Düşük eşik = Daha fazla enkaz tespiti (Ancak daha fazla hata olabilir).")
        line_width = st.slider("🖊️ Visualization Road Width", 2, 20, 6)
        
        st.markdown("#### Model Path")
        model_path = st.text_input("📁 Pytorch Weights (.pth)", value="models/en_iyi_model_cleaned.pth")
    
    analyze_btn = st.button("� Start Analysis", type="primary")
    if st.button("🔄 Reset Analysis"):
        st.session_state.analysis_result = None
        st.rerun()

# -----------------------------------------------------------------------------
# Main Interface
# -----------------------------------------------------------------------------

st.title("Satellite Road Damage Assessment")

# Map Display
m = folium.Map(location=city_coords, zoom_start=15)

# Add Base Layer
if provider == "Esri Wayback" and selected_wayback:
    url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{selected_wayback['id']}/{{z}}/{{y}}/{{x}}"
    folium.TileLayer(tiles=url, attr=f"Esri", name="Wayback").add_to(m)
elif provider == "Google Maps":
    folium.TileLayer(tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google", name="Google Satellite").add_to(m)
elif provider == "OpenAerialMap" and selected_oam_url:
    folium.TileLayer(tiles=selected_oam_url, attr="OAM", name="OAM Layer").add_to(m)

m.add_child(folium.LatLngPopup())
output = st_folium(m, height=450, width="100%")

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
        progress_bar = st.progress(0, text="Starting Analysis...")
        status_text = st.empty()
        
        try:
            # 1. Fetch Image based on clicked coordinate
            status_text.text("1/4 Downloading Satellite Imagery...")
            progress_bar.progress(10)
            
            wayback_id = selected_wayback['id'] if selected_wayback else None
            prov_code = 'google' if provider == "Google Maps" else 'custom' if provider == "OpenAerialMap" else 'esri'
            custom_url = selected_oam_url if provider == "OpenAerialMap" else None
            
            img, bounds = fetch_satellite_tile_custom(click_lat, click_lon, zoom_level, wayback_id=wayback_id, provider=prov_code, custom_url=custom_url)
            
            if img is None:
                st.error("FAILED to download imagery. Try a different zoom level or provider.")
                st.stop()
            else:
                progress_bar.progress(30)
                status_text.text("2/4 Fetching Road Network (OSM)...")
                
                # Image dims
                w, h = img.size
                
                # Overpass API road retrieval mapped onto exact bbox
                status_text.text("2/4 Fetching Road Network (OSM via Overpass)...")
                road_mask = get_osm_roads_overpass(bounds, w, h, thickness=line_width)
                road_mask_binary = (road_mask > 0).astype(np.uint8)

                status_text.text("3/4 Preprocessing & Model Loading...")
                progress_bar.progress(60)
                
                model, device = load_simple_model(model_path)
                if model is None:
                    st.error("Model yüklenemedi! Dosya yolunu kontrol edin.")
                    st.stop()
                    
                # 4. Inference
                status_text.text("4/4 Running AI Inference (Segformer)...")
                progress_bar.progress(80)
                
                img_np = np.array(img.convert("RGB"))
                
                # Resize if needed by the model
                input_tensor = cv2.resize(img_np, (512, 512))
                
                # PIL (RGB) formatı model tarafından doğru algılanması için float()'a çevrilirken standardize edildi
                input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float() / 255.0
                input_tensor = input_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                    
                # Resize prediction back to original image size (512x512)
                pred_mask = cv2.resize(pred_mask, (w, h))
                
                # Apply Neural Booster to enhance weak signals mapping slightly damaged regions
                pred_mask = np.clip(pred_mask * damage_booster, 0, 1)
                pred_mask_binary = (pred_mask > threshold).astype(np.uint8)
                
                # Intersection (Damage ON Road)
                intersection = cv2.bitwise_and(pred_mask_binary, road_mask_binary)
                
                progress_bar.progress(100)
                status_text.text("Analysis Complete!")
                
                st.session_state.analysis_result = {
                    "original_img": img_np,
                    "road_mask": road_mask_binary,
                    "pred_mask": pred_mask_binary,
                    "intersection": intersection,
                    "probs": pred_mask
                }
                st.rerun()
                
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.code(traceback.format_exc())
            progress_bar.empty()

# -----------------------------------------------------------------------------
# Result Display
# -----------------------------------------------------------------------------
if st.session_state.analysis_result:
    res = st.session_state.analysis_result
    
    # -- Canlı Eşikleme (Live Thresholding) --
    # Slider değiştikçe modeli baştan çalıştırmadan probabilities üzerinden anlık maske oluşturulur.
    current_pred_mask = (res["probs"] > threshold).astype(np.uint8)
    current_intersection = cv2.bitwise_and(current_pred_mask, res["road_mask"])
    
    st.subheader("📝 Analysis Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Satellite**")
        st.image(res["original_img"], width="stretch")
    
    with col2:
        st.markdown("**Damage Detection**")
        vis_img = res["original_img"].copy()
        
        # Color palettes
        yellow_overlay = np.zeros_like(vis_img)
        yellow_overlay[:] = [255, 255, 0] # Sarı (Genel Enkaz)
        
        red_overlay = np.zeros_like(vis_img)
        red_overlay[:] = [255, 0, 0] # Kırmızı (Yol üzeri Enkaz)
        
        cyan_overlay = np.zeros_like(vis_img)
        cyan_overlay[:] = [0, 255, 255] # Turkuaz (Açık yollar)
        
        # Saydam overlayleri resmin tamamı üzerinde oluşturup sonra maskeliyoruz (Index hatasını çözer)
        cyan_idx = (res["road_mask"] == 1) & (current_intersection == 0)
        blended_cyan = cv2.addWeighted(vis_img, 0.3, cyan_overlay, 0.7, 0)
        vis_img[cyan_idx] = blended_cyan[cyan_idx]
        
        mask_idx = (current_pred_mask == 1) & (current_intersection == 0)
        blended_yellow = cv2.addWeighted(vis_img, 0.5, yellow_overlay, 0.5, 0)
        vis_img[mask_idx] = blended_yellow[mask_idx]
        
        # Yol Üzerindeki Enkaz (Kesişim) - ÇOK DAHA BELİRGİN
        # Kesişim bölgesini daha çok şişiriyoruz
        kernel = np.ones((9, 9), np.uint8)
        thick_intersection = cv2.dilate(current_intersection, kernel, iterations=2)
        intersection_idx = (thick_intersection == 1)
        
        # Daha az saydam, daha opak bir kırmızı (0.1 baz, 0.9 kırmızı)
        blended_red = cv2.addWeighted(vis_img, 0.1, red_overlay, 0.9, 0)
        vis_img[intersection_idx] = blended_red[intersection_idx]

        st.image(vis_img, width="stretch")

    st.markdown("""
    <div style="background-color: #0E1117; padding: 15px; border-radius: 8px; border: 1px solid #333; display: flex; justify-content: space-around; margin-top: 10px;">
        <span style="color: #00FFFF; font-weight: bold; font-size: 1.1em;">■ Açık Yol (OSM)</span>
        <span style="color: #FFFF00; font-weight: bold; font-size: 1.1em;">■ Tespit Edilen Yıkıntı / Enkaz</span>
        <span style="color: #FF0000; font-weight: bold; font-size: 1.1em;">■ Yol Üzerindeki Yıkıntı (Kapalı Yol)</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧩 Diagnostik")
    c1, c2, c3 = st.columns(3)
    c1.image((current_pred_mask * 255).astype(np.uint8), caption="Model Yıkıntı Tahmini (Ham Maske)", width="stretch")
    c2.image((res["road_mask"] * 255).astype(np.uint8), caption="OSM Yol Ağı Maskesi", width="stretch")
    c3.image((current_intersection * 255).astype(np.uint8), caption="Yol & Yıkıntı Kesişimi", width="stretch")
