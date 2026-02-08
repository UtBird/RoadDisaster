import os
import sys
import argparse
import subprocess
import shutil
import json
import cv2
import numpy as np

# Add src to path just in case we need imports, but we use subprocess for inference
sys.path.append(os.path.join(os.getcwd(), "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model_dir", default="models", help="Directory containing downloaded model files")
    parser.add_argument("--output", default="output/prediction.jpg", help="Output image path")
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image {args.image} does not exist.")
        return

    # 1. Setup temporary directories
    temp_dir = "temp_inference"
    # unique temp dir to avoid conflicts? For demo, simple is fine.
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    imagery_dir = os.path.join(temp_dir, "imagery")
    os.makedirs(imagery_dir, exist_ok=True)
    
    # Copy input image to temp dir
    image_name = os.path.basename(args.image)
    target_image_path = os.path.join(imagery_dir, image_name)
    shutil.copy(args.image, target_image_path)
    
    # 2. Find model files
    ckpt_path = None
    yaml_path = None
    
    # specific structure created by download_model.py: models/CRASAR_RDA_UNet.../
    # We search recursively
    for root, dirs, files in os.walk(args.model_dir):
        for f in files:
            if f.endswith(".ckpt") and not ckpt_path:
                ckpt_path = os.path.join(root, f)
            if f.endswith(".yaml") and not yaml_path:
                yaml_path = os.path.join(root, f)
    
    if not ckpt_path or not yaml_path:
        print("Could not find .ckpt and .yaml files in " + args.model_dir)
        print("Please run download_model.py first.")
        return

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Using config: {yaml_path}")
    
    # 3. Create dummy folders for infer.py arguments
    spatial_dir = os.path.join(temp_dir, "spatial")
    adjustments_dir = os.path.join(temp_dir, "adjustments")
    boundaries_dir = os.path.join(temp_dir, "boundaries")
    os.makedirs(spatial_dir, exist_ok=True)
    os.makedirs(adjustments_dir, exist_ok=True)
    os.makedirs(boundaries_dir, exist_ok=True)
    
    preds_path = os.path.join(temp_dir, "preds.json")
    preds_folder = os.path.dirname(preds_path)
    os.makedirs(preds_folder, exist_ok=True)
    
    # 4. Run Inference
    # infer.py arguments
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
        # "--batch_size", "1" # Default is 2, 1 is safer
    ]
    
    # Add batch size if needed, but defaults might work. infer.py defaults to 2.
    # We only have 1 image.
    
    print("Running inference...")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with error code {e.returncode}")
        return
    
    # 5. Visualize
    if not os.path.exists(preds_path):
        print("Inference failed to produce predictions file.")
        return
        
    print("Inference finished. Visualizing...")
    try:
        with open(preds_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read predictions: {e}")
        return
        
    # Load original image
    img = cv2.imread(args.image)
    if img is None:
        print("Could not read original image for visualization.")
        return
        
    overlay = img.copy()
    
    # data["preds"] is a dictionary: road_line_id -> list of segment objects
    processed_preds = data.get("preds", {})
    
    if not processed_preds:
        print("No predictions found in output.")
    
    # RDA constants colors (Approximate based on earlier file view)
    DAMAGE_COLORS = {
        "No Damage": (0, 255, 0),       # Green
        "Minor Damage": (0, 255, 255),  # Yellow-ish (OpenCV uses BGR) -> Cyan? Yellow is (0, 255, 255) in BGR? No, Yellow is (0, 255, 255) in HSV. In BGR: Blue=0, Green=255, Red=255.
        "Major Damage": (0, 165, 255),  # Orangeb
        "Destroyed": (0, 0, 255),       # Red
        "Unclassified": (128, 128, 128)
    }
    
    # Verify colors BGR
    # Green: (0, 255, 0)
    # Yellow: (0, 255, 255)
    # Orange: (0, 127, 255)
    # Red: (0, 0, 255)
    
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for road_id, segments in processed_preds.items():
        if not segments:
            continue
            
        for segment in segments:
            label = segment.get("label", "Unclassified")
            pixels = segment.get("pixels", [])
            confidence = segment.get("confidence", 1.0)
            
            if pixels:
                # Convert list of {x, y} to list of [x, y]
                pts = np.array([[p["x"], p["y"]] for p in pixels], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                color = DAMAGE_COLORS.get(label, (128, 128, 128))
                
                # Draw the road segment
                # If it's a line string represented as a filled polygon or a thick line?
                # The data structure seems to imply it's a polygon (points defining shape).
                # `Spatial.py` calls it `pixel_geom` and `jsonify` dumps coords.
                # If it's a LineString, `fillPoly` might close it incorrectly.
                # `RoadLineFactory` uses `LineString`.
                # If it's a LineString, we should use `polylines`.
                # Wait, `RoadAnnotationPolygon` uses `Polygon`.
                # `LabeledRoadLine` inherits `RoadLine`.
                # `RoadLine` usually implies line.
                # However, RDA segments might include width?
                # In `BaseModelRDA`, `labeled_road_line_segments` are created with buffer width.
                # But `jsonify` dumps the geometry.
                # If `geometry` is `LineString`, we should use `polylines` with thickness.
                # If `geometry` is `Polygon` (buffered), we use `fillPoly`.
                
                # Let's check `road_lines_to_labeled_road_line_segments` in `decoder_utils.py`?
                # It likely returns LineStrings.
                # Let's assumethey are lines and draw them thick.
                
                cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=5)
                count += 1
                
                # Plot label (optional, might clutter)
                # x, y = pts[0][0]
                # cv2.putText(overlay, label, (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Add legend
    legend_x = 10
    legend_y = 30
    for label, color in DAMAGE_COLORS.items():
        cv2.rectangle(img, (legend_x, legend_y - 20), (legend_x + 20, legend_y), color, -1)
        cv2.putText(img, label, (legend_x + 30, legend_y - 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 30
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, img)
    print(f"Saved visualization to {args.output}. Drew {count} segments.")

if __name__ == "__main__":
    main()
