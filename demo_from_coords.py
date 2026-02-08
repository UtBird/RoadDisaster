import argparse
import os
import sys
import json
import shutil
import cv2
import numpy as np
from src.map_utils import fetch_satellite_tile, fetch_osm_roads, convert_roads_to_pixels

# Reuse run_demo logic or call it? 
# Better to import or replicate the relevant parts since run_demo was designed for a single image with no spatial data initially.
# But actually, run_demo.py calls infer.py. 
# We can prepare the data and then call infer.py exactly like run_demo.py does, 
# but this time we populate the spatial folder with real data!

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--zoom", type=int, default=19, help="Zoom level (19 is ~30cm/px)")
    parser.add_argument("--model_dir", default="models", help="Directory containing downloaded model files")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    args = parser.parse_args()

    # 1. Setup
    temp_dir = "temp_map_inference"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    imagery_dir = os.path.join(temp_dir, "imagery")
    spatial_dir = os.path.join(temp_dir, "spatial")
    adjustments_dir = os.path.join(temp_dir, "adjustments")
    boundaries_dir = os.path.join(temp_dir, "boundaries")
    
    os.makedirs(imagery_dir, exist_ok=True)
    os.makedirs(spatial_dir, exist_ok=True)
    os.makedirs(adjustments_dir, exist_ok=True)
    os.makedirs(boundaries_dir, exist_ok=True)
    
    # 2. Fetch Data
    print(f"Fetching satellite tile for {args.lat}, {args.lon} at zoom {args.zoom}...")
    img, bounds, tile = fetch_satellite_tile(args.lat, args.lon, args.zoom)
    
    if img is None:
        print("Failed to fetch satellite image.")
        return

    # Helper to save image
    img_filename = f"tile_{args.zoom}_{tile.x}_{tile.y}.jpg"
    img_path = os.path.join(imagery_dir, img_filename)
    img.convert("RGB").save(img_path)
    print(f"Saved image to {img_path}")
    
    print("Fetching OSM roads...")
    road_lines_latlon = fetch_osm_roads(bounds)
    print(f"Found {len(road_lines_latlon)} road segments.")
    
    # Convert to pixels
    road_pixels = convert_roads_to_pixels(road_lines_latlon, bounds, img.width, img.height)
    
    # Save spatial JSON
    # The filename MUST match the image filename (except extension) for the loader to find it?
    # Orthomosaic.py Line 436: `buildings_file = find_geotif_file_prefix_match(target_file, ...)`
    # It matches prefix. So "tile_... .json" should work for "tile_... .jpg"
    
    json_filename = img_filename.replace(".jpg", ".json")
    json_path = os.path.join(spatial_dir, json_filename)
    
    road_line_objs = []
    for i, pixels in enumerate(road_pixels):
        # Format as expected by Spatial.py
        # "pixels": [{"x":..., "y":...}, ...]
        formatted_pixels = [{"x": p["x"], "y": p["y"]} for p in pixels]
        
        road_line_objs.append({
            "id": f"road_{i}",
            "label": "Road Line",
            "pixels": formatted_pixels,
            "EPSG:4326": [], # Empty for now
            "source": "OSM"
        })
        
    json_data = {
        "road_lines": road_line_objs,
        "polygons": []
    }
    
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
        
    print(f"Saved spatial data to {json_path}")
    
    # 3. Find Model
    ckpt_path = None
    yaml_path = None
    for root, dirs, files in os.walk(args.model_dir):
        for f in files:
            if f.endswith(".ckpt") and not ckpt_path:
                ckpt_path = os.path.join(root, f)
            if f.endswith(".yaml") and not yaml_path:
                yaml_path = os.path.join(root, f)
                
    if not ckpt_path or not yaml_path:
        print("Model files not found.")
        return

    # 4. Run Inference
    import subprocess
    preds_path = os.path.join(temp_dir, "preds.json")
    
    cmd = [
        sys.executable,
        "src/modeling/infer.py",
        "--test_imagery_folder", imagery_dir,
        "--test_spatial_folder", spatial_dir, 
        "--test_adjustments_folder", adjustments_dir,
        "--test_boundaries_folder", boundaries_dir,
        "--hyperparameters_yaml_path", yaml_path,
        "--model_path", ckpt_path,
        "--preds_path", preds_path,
        "--data_gen_workers", "0", 
        "--accelerator", "cpu", 
        "--batch_size", "1"
    ]
    
    print("Running inference...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Inference failed: {e}")
        return

    # 5. Visualize
    print("Visualizing...")
    if not os.path.exists(preds_path):
        print("No predictions found.")
        return
        
    with open(preds_path, "r") as f:
        preds_data = json.load(f)
        
    # Re-read image using cv2
    cv_img = cv2.imread(img_path)
    overlay = cv_img.copy()
    
    processed_preds = preds_data.get("preds", {})
    
    DAMAGE_COLORS = {
        "No Damage": (0, 255, 0),       # Green
        "Minor Damage": (0, 255, 255),  # Yellow
        "Major Damage": (0, 165, 255),  # Orange
        "Destroyed": (0, 0, 255),       # Red
        "Unclassified": (128, 128, 128)
    }
    
    count = 0
    # Also draw OSM roads in Blue for reference?
    # for road in road_pixels:
    #     pts = np.array([[p["x"], p["y"]] for p in road], np.int32).reshape((-1, 1, 2))
    #     cv2.polylines(overlay, [pts], False, (255, 0, 0), 1)
    
    for road_id, segments in processed_preds.items():
        if not segments:
            continue
        for segment in segments:
            label = segment.get("label", "Unclassified")
            pixels = segment.get("pixels", [])
            
            if pixels:
                pts = np.array([[p["x"], p["y"]] for p in pixels], np.int32).reshape((-1, 1, 2))
                color = DAMAGE_COLORS.get(label, (128, 128, 128))
                cv2.polylines(overlay, [pts], False, color, 4)
                count += 1

    # Blend
    cv2.addWeighted(overlay, 0.6, cv_img, 0.4, 0, cv_img)
    
    # Legend
    legend_x = 10
    legend_y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, color in DAMAGE_COLORS.items():
        cv2.rectangle(cv_img, (legend_x, legend_y - 20), (legend_x + 20, legend_y), color, -1)
        cv2.putText(cv_img, label, (legend_x + 30, legend_y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 25
        
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"map_result_{args.lat}_{args.lon}.jpg")
    cv2.imwrite(output_path, cv_img)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    main()
