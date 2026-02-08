import os
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

def download_model():
    api = HfApi()
    
    print("Searching for RDA models in CRASAR organization...")
    models = []
    try:
        models = list(api.list_models(author="CRASAR", search="RDA"))
    except Exception as e:
        print(f"Warning: Failed to list models: {e}")
        # Fallback will handle this
    
    # Filter for UNet if possible, or just take the first relevant one
    target_model_id = None
    if models:
        for model in models:
            print(f"Found: {model.modelId}")
            if "UNet" in model.modelId and "RDA" in model.modelId:
                target_model_id = model.modelId
                break
    
    if not target_model_id:
        print("Could not find a specific RDA UNet model via search. Using hardcoded fallback.")
        target_model_id = "CRASAR/RDA_UNet_full_attention_CCE_full-v1" 
        # Note: I am guessing the ID based on the paper/repo. 
        # But wait, looking at the batch file again: 
        # "RDA_UNet_full_attention_CCE_full" might be a folder name.
        # Let's try searching for "RDA_UNet" in public models generally if CRASAR fails?
        # No, CRASAR is the org.
        # If list fails, I'll try to guess a few IDs or ask user.
        # Let's try "CRASAR/RDA_UNet_full_v1" or similar. 
        # Actually, let's try to just download from the COLLECTION if possible? No.
        pass

    if not target_model_id:
         target_model_id = "CRASAR/RDA_UNet_full_v1" # Another guess


    print(f"Targeting model: {target_model_id}")
    
    # Download the model checkpoints and yaml configs
    # We need the specific .ckpt and .yaml used in the paper/repo
    print("Downloading model files...")
    local_dir = os.path.join("models", target_model_id.replace("/", "_"))
    os.makedirs(local_dir, exist_ok=True)
    
    # Download everything or just specific patterns
    # The batch file used: RDA_UNet_full_v1-epoch=02-step=1500-val_macro_iou=0.07566.ckpt
    try:
        path = snapshot_download(
            repo_id=target_model_id,
            allow_patterns=["*.ckpt", "*.yaml", "*.json"],
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded to: {path}")
        
        # List downloaded files to help user
        files = os.listdir(path)
        print("Downloaded files:", files)
        
        # heuristics to find the right files
        ckpt = [f for f in files if f.endswith(".ckpt")][0] if any(f.endswith(".ckpt") for f in files) else None
        config = [f for f in files if f.endswith(".yaml")][0] if any(f.endswith(".yaml") for f in files) else None
        
        if ckpt and config:
            print(f"Found checkpoint: {ckpt}")
            print(f"Found config: {config}")
        else:
            print("Warning: Could not identify .ckpt or .yaml files automatically.")
            
    except Exception as e:
        print(f"Failed to download from {target_model_id}: {e}")
        print("Please check the model ID or internet connection.")

if __name__ == "__main__":
    download_model()
