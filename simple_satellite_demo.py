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

def fetch_satellite_tile(lat, lon, zoom=19, wayback_id=None):
    """Fetch Esri World Imagery tile. Supports Wayback if ID is provided."""
    tile = mercantile.tile(lon, lat, zoom)
    
    if wayback_id:
        # Esri Wayback URL pattern: .../tile/{wayback_id}/{z}/{y}/{x}
        url = f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{wayback_id}/{zoom}/{tile.y}/{tile.x}"
    else:
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile.y}/{tile.x}"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        print(f"Fetched tile: {zoom}/{tile.x}/{tile.y} (Wayback: {wayback_id})")
        return img, mercantile.bounds(tile)
    except Exception as e:
        print(f"Error fetching tile: {e}")
        return None, None

def fetch_osm_line_pixels(bounds, width, height):
    """Fetch OSM roads and convert to list of pixel polygons (thick lines)."""
    # Overpass API Query
    # [out:json][timeout:25];
    # (
    #   way["highway"](south, west, north, east);
    # );
    # (._;>;);
    # out body;
    
    bbox_str = f"{bounds.south},{bounds.west},{bounds.north},{bounds.east}"
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"]({bbox_str});
    );
    (._;>;);
    out body;
    """
    
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter"
    ]
    
    data = None
    for endpoint in endpoints:
        try:
            print(f"Fetching OSM data from {endpoint}...")
            r = requests.post(endpoint, data={'data': query}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                break
            else:
                print(f"Failed with {endpoint}: Status {r.status_code}")
        except Exception as e:
            print(f"Failed with {endpoint}: {e}")
            
    if not data:
        print("All Overpass servers failed.")
        return []
        
    # Process JSON
    # 1. Build Node Map
    nodes = {}
    if 'elements' in data:
        for el in data['elements']:
            if el['type'] == 'node':
                nodes[el['id']] = (el['lon'], el['lat'])
                
        # 2. Build Ways
        road_polygons = []
        for el in data['elements']:
            if el['type'] == 'way' and 'nodes' in el:
                points = []
                for node_id in el['nodes']:
                    if node_id in nodes:
                        lon, lat = nodes[node_id]
                        
                        # Convert to Pixels
                        px = (lon - bounds.west) / (bounds.east - bounds.west) * width
                        py = (bounds.north - lat) / (bounds.north - bounds.south) * height
                        points.append([int(px), int(py)])
                
                if len(points) >= 2:
                    road_polygons.append(np.array(points, dtype=np.int32))
                    
        return road_polygons
    return []

def parse_dms(dms_str):
    """Parse DMS string (e.g. 36°13'38\\"N) to decimal degrees."""
    import re
    # Simple regex for 36°13'38"N
    # Adjust regex to be flexible
    pattern = re.compile(r"(\d+)°(\d+)'(\d+(?:\.\d+)?)\"([NSEW])")
    match = pattern.search(dms_str)
    if not match:
        try:
            return float(dms_str)
        except ValueError:
            print(f"Could not parse coordinate: {dms_str}")
            return None
            
    deg, min, sec, direction = match.groups()
    val = float(deg) + float(min)/60 + float(sec)/3600
    if direction in ['S', 'W']:
        val *= -1
    return val

def main():
    print("Script started. Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", required=True, help="Latitude (Decimal or DMS)")
    parser.add_argument("--lon", required=True, help="Longitude (Decimal or DMS)")
    parser.add_argument("--zoom", type=int, default=19)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="output/simple_result.jpg")
    parser.add_argument("--date", help="(Experimental) specific date string. Note: Public APIs often don't support historical queries.")
    args = parser.parse_args()
    
    print(f"Received arguments: lat='{args.lat}', lon='{args.lon}', zoom={args.zoom}")
    
    lat = parse_dms(args.lat)
    lon = parse_dms(args.lon)
    
    if lat is None or lon is None:
        print("Error: Could not parse latitude or longitude.")
        return

    print(f"Analyzing location: {lat}, {lon}")
    if args.date:
        print(f"Note: Historical imagery for {args.date} requested. Using latest available Esri World Imagery. "
              "For specific dates, a paid provider (Planet/Maxar) API is usually required.")

    # 1. Fetch Image
    img, bounds = fetch_satellite_tile(lat, lon, args.zoom)
    if img is None: return
    
    w, h = img.size
    
    # 2. Fetch Roads & Create Mask
    road_polygons = fetch_osm_line_pixels(bounds, w, h)
    
    # Create mask: 1 where road, 0 elsewhere
    # Model expects mask in channel 3 (0-indexed -> 4th channel)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw roads on mask. 
    # RDA Road width in pixels? Config says "road_line_buffer_width_pixels: 40". 
    # We should draw lines with similar thickness (40px) to simulate the mask expected by training.
    for poly in road_polygons:
        cv2.polylines(mask, [poly], False, 1, thickness=40)
        
    # 3. Prepare Input Tensor
    # Order: Red, Green, Blue, Mask (based on config: blue:2, green:1, mask:3, red:0 ?? No.)
    # Config says: 
    # channels:
    #   blue: 2
    #   green: 1
    #   mask: 3
    #   red: 0
    # This usually means Input Tensor channels are [Red, Green, Blue, Mask].
    # And the indices map to source data? 
    # Or does it mean Channel 0 of input is Red, Channel 1 is Green...?
    # Standard image is RGB. 
    # Let's assume input tensor is (Batch, 4, H, W) where 0=R, 1=G, 2=B, 3=Mask.
    
    img_np = np.array(img) # H, W, 3 (RGB)
    
    # Normalize 0-1
    img_float = img_np.astype(np.float32) / 255.0
    mask_float = mask.astype(np.float32) # Already 0-1 if we drew with 1
    
    # Stack: (H, W, 4)
    input_data = np.dstack([img_float, mask_float])
    
    # Transpose to (C, H, W) -> (4, H, W)
    input_tensor = torch.from_numpy(input_data).permute(2, 0, 1).unsqueeze(0) # (1, 4, H, W)
    
    # 4. Load Model
    print("Loading model...")
    # n_channels=4, n_classes=4
    # input_channel_mask_index=3 (The 4th channel is the mask)
    # output_channel_background_index=0 (Background class is 0)
    model = MaskedUNet(n_channels=4, n_classes=4, hyperparameters=HYPERPARAMETERS, 
                       input_channel_mask_index=3, output_channel_background_index=0)
    
    try:
        ckpt = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict_from_ckpt(ckpt["state_dict"])
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
        
    model.eval()
    
    # 5. Inference
    print("Running inference...")
    with torch.no_grad():
        # returns (1, 4, H, W)
        preds = model(input_tensor)
        # argmax -> (1, H, W)
        pred_labels = torch.argmax(preds, dim=1).squeeze().numpy()
        
    # 6. Visualize
    # Map classes to colors
    # 0: Background (Transparent)
    # 1: Road Line (Green)
    # 2: Partial (Yellow)
    # 3: Total (Red)
    
    vis_img = img_np.copy()
    
    # Create overlays
    green = np.zeros_like(vis_img)
    green[:] = [0, 255, 0] # RGB
    yellow = np.zeros_like(vis_img)
    yellow[:] = [255, 255, 0]
    red = np.zeros_like(vis_img)
    red[:] = [255, 0, 0]
    
    # Apply where masks match
    # Only color pixels where we also have the Road Mask? 
    # The model should already be masking output (Maskable) so background should be 0 where no road.
    
    road_mask = (pred_labels == 1)
    partial_mask = (pred_labels == 2)
    total_mask = (pred_labels == 3)
    
    alpha = 0.5
    
    # Create a full color overlay
    overlay = vis_img.copy()
    
    # Paint colors on the overlay
    overlay[road_mask] = [0, 255, 0]      # Green
    overlay[partial_mask] = [255, 255, 0] # Yellow
    overlay[total_mask] = [255, 0, 0]     # Red
    
    # Blend the overlay with the original
    # This blurs everything slightly if alpha < 1, but highlights damages
    # Better: Only blend where mask is true?
    # cv2.addWeighted blends everything.
    
    # Create a combined mask of all interesting pixels
    combined_mask = road_mask | partial_mask | total_mask
    
    # Blend only on the combined mask
    if combined_mask.any():
        # Blended chunk
        blended = cv2.addWeighted(vis_img, 1-alpha, overlay, alpha, 0)
        # Apply only to interesting pixels
        vis_img[combined_mask] = blended[combined_mask]
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Image.fromarray(vis_img).save(args.output)
    print(f"Saved result to {args.output}")

if __name__ == "__main__":
    main()
