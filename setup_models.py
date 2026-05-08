import os
import requests
from tqdm import tqdm

MODELS = {
    "yolo_best.pt": "https://huggingface.co/siwon23/violence-detection-yolov8/resolve/main/yolo_best.pt",
    "weapon_yolov8.pt": "https://huggingface.co/Hadi959/weapon-detection-yolov8/resolve/main/best.pt",
    "yolov8n-blood-violence.pt": "https://huggingface.co/Notacodinggeek/yolov8n-blood-violence/resolve/main/yolov8n-blood-violence.pt"
}

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def setup_models(model_dir="models"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    for filename, url in MODELS.items():
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            download_file(url, filepath)
        else:
            print(f"{filename} already exists.")

if __name__ == "__main__":
    setup_models()
