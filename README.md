# Satellite Road Damage Assessment (RDA)

A simplified tool to detect road damage (cracks, flooding, debris) from satellite imagery using a pre-trained UNet model (CRASAR RDA).

## Features
- **Map Interface**: Select any location on earth via Streamlit UI.
- **Historical Imagery**: Access Esri Wayback archives to analyze past events (e.g., Feb 2023 Earthquake).
- **Road Masking**: Automatically fetches OSM road data to focus analysis on drivable surfaces.
- **Damage Visualization**: Overlays damage predictions (Green=Safe, Yellow=Partial, Red=Destroyed).

## Installation

1.  **Clone the repository**
2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/Mac
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Model Setup
The model file (~930MB) is **not included** in this repository due to size limits.
You must download it separately:

1.  **Run the download script**:
    ```bash
    python download_model.py
    ```
    *This will fetch the `RDA_UNet_simple_noattention.ckpt` from Hugging Face and place it in `models/`.*

## Usage

### 1. Web Interface (Recommended)
Run the Streamlit app:
```bash
streamlit run app.py
```
- Opens a web UI.
- Select a location on the map.
- Choose a historical date (optional).
- Click **Analyze**.

### 2. Command Line (Advanced)
Run analysis on specific coordinates:
```bash
python simple_satellite_demo.py --lat 36.227222 --lon 36.165555 --zoom 18 --model_path models/CRASAR.../model.ckpt
```

## Credits
Based on the CRASAR / RDA UNet research.
