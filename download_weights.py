import os
import urllib.request
from pathlib import Path

Path("weights").mkdir(exist_ok=True)

# 1. AV-HuBERT (Base VSR Model from Meta)
AVHUBERT_URL = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt"
AVHUBERT_PATH = "weights/avhubert_base_vsr.pt"

# 2. LipNet (PyTorch Weights hosted on Hugging Face by the SilentSpeak team)
LIPNET_URL = "https://huggingface.co/singhhrishabh/silentassist-lipnet-grid/resolve/main/silentassist_lipnet_grid.pt"
LIPNET_PATH = "weights/lipnet.pt"

def download_file(url, dest_path, model_name):
    if os.path.exists(dest_path):
        print(f"{model_name} weights already exist at {dest_path}")
        return

    print(f"Downloading {model_name} weights (this may take a few minutes)...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Successfully downloaded {model_name} to {dest_path}")
    except Exception as e:
        print(f"Failed to download {model_name}. Error: {e}")

if __name__ == "__main__":
    print("--- SafeSpeak Weights Downloader ---")
    download_file(AVHUBERT_URL, AVHUBERT_PATH, "AV-HuBERT")
    download_file(LIPNET_URL, LIPNET_PATH, "LipNet")
    print("--- Done ---")
