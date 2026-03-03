import argparse
import os
import sys
import torch
import cv2
import numpy as np
import mercantile
import requests
from io import BytesIO
from PIL import Image

from simple_model_defs import MaskedUNet

# Hardcoded Hyperparameters to match RDA_UNet_simple_noattention_adjusted.yaml
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
                "up_0": {"attention": False, "bilinear": False, "in_channels": 1024, "out_channels": 512},
                "up_1": {"attention": False, "bilinear": False, "in_channels": 512, "out_channels": 256},
                "up_2": {"attention": False, "bilinear": False, "in_channels": 256, "out_channels": 128},
                "up_3": {"attention": False, "bilinear": False, "in_channels": 128, "out_channels": 64},
                "up_4": {"attention": False, "bilinear": False, "in_channels": 64, "out_channels": 32}
            }
        }
    }
}

def fetch_satellite_tile(lat, lon, zoom=19, wayback_id=None, provider='esri', custom_url=None):
    """Fetch Esri World Imagery tile. Supports Wayback if ID is provided."""
    tile = mercantile.tile(lon, lat, zoom)
    
    if provider == 'google':
        url = f"https://mt1.google.com/vt/lyrs=s&x={tile.x}&y={tile.y}&z={zoom}"
    elif provider == 'custom' and custom_url:
         url = custom_url.replace("{x}", str(tile.x)).replace("{y}", str(tile.y)).replace("{z}", str(zoom))
    elif wayback_id:
        # Esri Wayback URL pattern: .../tile/{wayback_id}/{z}/{y}/{x}
        url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{wayback_id}/{zoom}/{tile.y}/{tile.x}"
    else:
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile.y}/{tile.x}"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        print(f"DEBUG: Fetching {url}")
        r = requests.get(url, headers=headers, timeout=10) # 10s timeout
        if r.status_code != 200:
             print(f"DEBUG: Failed with status {r.status_code}")
             return None, None
             
        img = Image.open(BytesIO(r.content)).convert("RGB")
        print(f"Fetched tile: {zoom}/{tile.x}/{tile.y}")
        return img, mercantile.bounds(tile)
    except Exception as e:
        print(f"Error fetching tile: {e}")
        return None, None

def fetch_osm_line_pixels(bounds, width, height):
    """Fetch OSM roads using optimized 'out geom' for maximum speed and stability."""
    
    bbox_str = f"{bounds.south},{bounds.west},{bounds.north},{bounds.east}"
    # OPTIMIZATION: Use 'out geom' instead of recursion. Much lighter on servers.
    query = f"""
    [out:json][timeout:60];
    way["highway"]({bbox_str});
    out geom;
    """
    
    # 2025 High-Availability Mirrors & Robust Rotation
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
        "https://overpass.openstreetmap.fr/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.nchc.org.tw/api/interpreter",
        "https://overpass.osm.ch/api/interpreter",
        "https://overpass.be/api/interpreter"
    ]
    
    import random
    import time
    random.shuffle(endpoints)
    
    # Realistic User-Agent to avoid 403s
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RoadDamageApp/1.2",
        "Referer": "https://overpass-turbo.eu/"
    }
    
    data = None
    for endpoint in endpoints:
        for attempt in range(2): 
            try:
                print(f"Fetching OSM from {endpoint}...")
                # Increased timeout to 60s for heavy post-disaster areas
                r = requests.post(endpoint, data={'data': query}, headers=headers, timeout=60)
                
                if r.status_code == 200:
                    data = r.json()
                    if 'elements' in data and len(data['elements']) > 0:
                        break 
                elif r.status_code == 429 or r.status_code >= 500:
                    print(f"Server busy ({r.status_code}). Retrying in 3s...")
                    time.sleep(3)
                    continue 
                else:
                    print(f"Failed with {endpoint}: Status {r.status_code}")
                    break 
            except Exception as e:
                print(f"Connection error on {endpoint}: {e}")
                time.sleep(1) # Small pause before next mirror
                break 
        if data: break
            
    if not data or 'elements' not in data:
        print("CRITICAL: All Overpass API methods failed. Continuing without mask.")
        return []
        
    # Process Optimized JSON (out geom contains lat/lon inside way)
    road_polygons = []
    for el in data.get('elements', []):
        if el['type'] == 'way' and 'geometry' in el:
            points = []
            for pt in el['geometry']:
                lon, lat = pt['lon'], pt['lat']
                px = (lon - bounds.west) / (bounds.east - bounds.west) * width
                py = (bounds.north - lat) / (bounds.north - bounds.south) * height
                points.append([int(px), int(py)])
            
            if len(points) >= 2:
                road_polygons.append(np.array(points, dtype=np.int32))
                
    return road_polygons

def parse_dms(dms_str):
    import re
    pattern = re.compile(r"(\d+)°(\d+)'(\d+(?:\.\d+)?)\"([NSEW])")
    match = pattern.search(dms_str)
    if not match:
        try:
            return float(dms_str)
        except ValueError:
            return None
            
    deg, min, sec, direction = match.groups()
    val = float(deg) + float(min)/60 + float(sec)/3600
    if direction in ['S', 'W']:
        val *= -1
    return val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", required=True)
    parser.add_argument("--lon", required=True)
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="output/simple_result.jpg")
    parser.add_argument("--provider", default="google")
    args = parser.parse_args()
    
    lat = parse_dms(args.lat)
    lon = parse_dms(args.lon)
    
    if lat is None or lon is None:
        print("Invalid coords")
        return

    img, bounds = fetch_satellite_tile(lat, lon, args.zoom, provider=args.provider)
    if img is None: return
    
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
    
    model = MaskedUNet(n_channels=4, n_classes=4, hyperparameters=HYPERPARAMETERS, 
                       input_channel_mask_index=3, output_channel_background_index=0)
    
    ckpt = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict_from_ckpt(ckpt["state_dict"])
    model.eval()
    
    with torch.no_grad():
        preds = model(input_tensor)
        pred_labels = torch.argmax(preds, dim=1).squeeze().numpy()
        
    vis_img = img_np.copy()
    overlay = vis_img.copy()
    
    overlay[pred_labels == 1] = [0, 255, 0]
    overlay[pred_labels == 2] = [255, 255, 0]
    overlay[pred_labels == 3] = [255, 0, 0]
    
    combined_mask = (pred_labels > 0)
    alpha = 0.5
    
    if combined_mask.any():
        blended = cv2.addWeighted(vis_img, 1-alpha, overlay, alpha, 0)
        vis_img[combined_mask] = blended[combined_mask]
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Image.fromarray(vis_img).save(args.output)
    print(f"Saved result to {args.output}")

if __name__ == "__main__":
    main()
