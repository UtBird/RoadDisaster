import os
import math
import requests
import mercantile
import json
from PIL import Image
from io import BytesIO
from OSMPythonTools.overpass import overpassQueryBuilder, Overpass

def latlon_to_tile(lat, lon, zoom):
    """
    Converts lat/lon to tile coordinates (x, y).
    """
    return mercantile.tile(lon, lat, zoom)

def fetch_satellite_tile(lat, lon, zoom=19, provider='esri', wayback_id=None, custom_url=None):
    """
    Fetches a satellite tile from a public XYZ source (Esri World Imagery).
    Returns the PIL Image and the tile bounds (west, south, east, north).
    """
    tile = latlon_to_tile(lat, lon, zoom)
    
    # Esri World Imagery (ArcGIS) - varying availability but generally good
    # url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile.y}/{tile.x}"
    
    # Alternative: Google Satellite (Requires key usually, but mt1.google.com sometimes works for testing)
    # Let's stick to Esri for now as it is often used in open mapping.
    # Alternative: Google Satellite (mt1.google.com)
    # url = f"https://mt1.google.com/vt/lyrs=s&x={tile.x}&y={tile.y}&z={zoom}"
    
    if provider == 'google':
        url = f"https://mt1.google.com/vt/lyrs=s&x={tile.x}&y={tile.y}&z={zoom}"
    elif provider == 'custom' and custom_url:
        # Expecting {x}, {y}, {z} in the URL
        url = custom_url.replace("{x}", str(tile.x)).replace("{y}", str(tile.y)).replace("{z}", str(zoom))
    else:
        # Default to Esri
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{tile.y}/{tile.x}"
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        
        bounds = mercantile.bounds(tile)
        return img, bounds, tile
    except Exception as e:
        print(f"Failed to fetch tile: {e}")
        return None, None, None

def fetch_osm_roads(bounds):
    """
    Fetches road lines within the given bounds (west, south, east, north) using Overpass API.
    Returns a list of road LineStrings in lat/lon coordinates.
    """
    # bounds: (west, south, east, north)
    # Overpass expects (south, west, north, east)
    bbox = (bounds.south, bounds.west, bounds.north, bounds.east)
    
    overpass = Overpass()
    query = overpassQueryBuilder(bbox=bbox, elementType='way', selector='"highway"', out='body')
    
    try:
        result = overpass.query(query)
        roads = result.ways()
        
        road_lines = []
        for road in roads:
            # Get geometry (nodes)
            nodes = road.nodes()
            points = [(float(node.lat()), float(node.lon())) for node in nodes]
            if len(points) >= 2:
                road_lines.append(points)
        
        return road_lines
    except Exception as e:
        print(f"Failed to fetch OSM roads: {e}")
        return []

def convert_roads_to_pixels(road_lines, bounds, img_width, img_height):
    """
    Converts road lines from lat/lon to pixel coordinates relative to the tile bounds.
    """
    west, south, east, north = bounds.west, bounds.south, bounds.east, bounds.north
    
    pixel_roads = []
    
    # Simple linear interpolation for small tiles (Spherical Mercator is linear in web mercator, 
    # but for small deviations lat/lon is "okay" ish, but better to use projection. 
    # Since we are using mercantile, we should ideally project to meters first. 
    # However, for a single tile at zoom 19, linear interpolation on lat/lon is often "close enough" 
    # because the tile is small. But let's be slightly more robust if we can.
    # Actually, mercantile bounds are in Lat/Lon. The tile is 256x256 pixels.
    # We can use the ratio.
    
    lat_span = north - south
    lon_span = east - west
    
    for road in road_lines:
        pixels = []
        for lat, lon in road:
            # Normalize to 0-1
            # Note: Y axis in image is top-down (0 is north).
            # Lat: North is max, South is min. So 0px is North.
            # px_y = (north - lat) / lat_span * height
            
            # Lon: West is min, East is max. So 0px is West.
            # px_x = (lon - west) / lon_span * width
            
            # Check if point is roughly within or near the tile
            # We allow some margin?
            
            px_x = (lon - west) / lon_span * img_width
            px_y = (north - lat) / lat_span * img_height
             
            # Clip? Or just include? The drawing will clip.
            pixels.append({"x": px_x, "y": px_y})
            
        pixel_roads.append(pixels)
        
    return pixel_roads

def save_data_for_inference(img, pixel_roads, output_dir, file_prefix="map_sample"):
    """
    Saves the image and creates the JSON structure expected by infer.py/Spatial.py.
    """
    # 1. Save Image
    os.makedirs(os.path.join(output_dir, "imagery"), exist_ok=True)
    img_path = os.path.join(output_dir, "imagery", f"{file_prefix}.jpg")
    img.convert("RGB").save(img_path)
    
    # 2. Create Spatial JSON
    # Structure based on src/modeling/Spatial.py and samples
    # It expects:
    # {
    #   "road_lines": [ { "id": "...", "label": "Road Line", "pixels": [{x,y},...], "EPSG:4326": ... } ],
    #   "polygons": [ ... ] (Optional/Empty for inference typically?)
    # }
    
    # Wait, infer.py loads "spatial" using `RoadLineFactory` from `rda_annotations["road_lines"]`? 
    # Or `bda_annotation_folder`?
    # In infer.py lines 104-106:
    # if hyperparams["task"] == "RDA":
    #     test_rda_labels_path = args.test_spatial_folder
    # Line 116: rda_annotation_folder=test_rda_labels_path
    
    # Orthomosaic.py Line 453:
    # if rda_annotation_folder:
    #     rda_file = find_geotif_file_prefix_match(target_file, ...)
    #     json.load(...)
    #     road_lines = RoadLineFactory(rda_annotations["road_lines"])
    
    # So we need a JSON file named similarly to the image (e.g., map_sample.json) in the spatial folder.
    # It should have a key "road_lines".
    
    road_line_objs = []
    for i, road_pixels in enumerate(pixel_roads):
        road_line_objs.append({
            "id": f"road_{i}",
            "label": "Road Line",
            "pixels": road_pixels,
            "EPSG:4326": [], # Optional if we don't need it for inference logic?
            "source": "OSM"
        })
        
    json_data = {
        "road_lines": road_line_objs,
        "polygons": []
    }
    
    os.makedirs(os.path.join(output_dir, "spatial"), exist_ok=True)
    json_path = os.path.join(output_dir, "spatial", f"{file_prefix}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
        
    return img_path, json_path

def search_oam_images(bbox, date_start=None, date_end=None, limit=50):
    """
    Search OpenAerialMap for images within a bbox and date range.
    bbox: (west, south, east, north)
    date_start, date_end: datetime objects or YYYY-MM-DD strings
    """
    url = "https://api.openaerialmap.org/meta"
    
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "limit": limit,
        "order_by": "acquisition_end",
        "sort": "desc"
    }
    
    if date_start and date_end:
        # OAM expects timestamp range usually? Or just filter results.
        # The API documentation says 'acquisition_from' and 'acquisition_to' might work, 
        # or we verify the results.
        # Let's try 'acquisition_from' and 'acquisition_to' based on common OAM usage.
        if isinstance(date_start, str): params["acquisition_from"] = date_start
        else: params["acquisition_from"] = date_start.strftime("%Y-%m-%d")
            
        if isinstance(date_end, str): params["acquisition_to"] = date_end
        else: params["acquisition_to"] = date_end.strftime("%Y-%m-%d")

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        results = []
        if 'results' in data:
            for item in data['results']:
                # We need the tile URL. usually in 'properties' -> 'tms' or 'wmts'
                props = item.get('properties', {}) or {} # Sometimes properties is None?
                # Actually OAM API response structure:
                # { results: [ { uuid, title, properties: { tms, ... }, bbox, ... } ] }
                
                tms_url = item.get('tms') or item.get('properties', {}).get('tms')
                if not tms_url:
                    # Sometimes provided as 'wmts'
                    tms_url = item.get('wmts') or item.get('properties', {}).get('wmts')
                
                if tms_url:
                    results.append({
                        "id": item.get('_id') or item.get('uuid'),
                        "title": item.get('title', 'Unknown Image'),
                        "provider": item.get('provider', 'Unknown'),
                        "date": item.get('acquisition_end', item.get('acquisition_start', 'Unknown Date')),
                        "resolution": item.get('resolution', 'N/A'),
                        "tms_url": tms_url,
                        "bbox": item.get('bbox')
                    })
        return results
    except Exception as e:
        print(f"OAM Search failed: {e}")
        return []


if __name__ == "__main__":
    # Test
    # Lat/Lon near the user or a known road? 
    # Let's try a safe default
    pass
