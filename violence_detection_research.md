# Research Report: Specific Violence and Weapon Detection Models

## 1. Recommended Pre-trained YOLOv8 Models

The following models are trained on YOLOv8 and can be easily integrated using the `ultralytics` Python library.

### A. General Violence Detection
*   **Model**: `siwon23/violence-detection-yolov8`
*   **Weights File**: `yolo_best.pt`
*   **Hugging Face Link**: [https://huggingface.co/siwon23/violence-detection-yolov8](https://huggingface.co/siwon23/violence-detection-yolov8)
*   **Download Instructions**:
    ```bash
    wget https://huggingface.co/siwon23/violence-detection-yolov8/resolve/main/yolo_best.pt
    ```

### B. Weapon Detection (Guns, Knives)
*   **Model**: `Hadi959/weapon-detection-yolov8`
*   **Weights File**: `best.pt`
*   **Hugging Face Link**: [https://huggingface.co/Hadi959/weapon-detection-yolov8](https://huggingface.co/Hadi959/weapon-detection-yolov8)
*   **Download Instructions**:
    ```bash
    wget https://huggingface.co/Hadi959/weapon-detection-yolov8/resolve/main/best.pt -O weapon_yolov8.pt
    ```

### C. Blood and Violence Classification
*   **Model**: `Notacodinggeek/yolov8n-blood-violence`
*   **Weights File**: `yolov8n-blood-violence.pt`
*   **Hugging Face Link**: [https://huggingface.co/Notacodinggeek/yolov8n-blood-violence](https://huggingface.co/Notacodinggeek/yolov8n-blood-violence)
*   **Download Instructions**:
    ```bash
    wget https://huggingface.co/Notacodinggeek/yolov8n-blood-violence/resolve/main/yolov8n-blood-violence.pt
    ```

## 2. Advanced Action Recognition (Temporal Analysis)

While YOLOv8 works on individual frames, temporal models are better at detecting "fighting" as a sequence of actions.

*   **Model**: `Myaukko/videomae-base-finetuned-fighting`
*   **Architecture**: VideoMAE (Video Masked Autoencoder)
*   **Hugging Face Link**: [https://huggingface.co/Myaukko/videomae-base-finetuned-fighting-subset-5ep-5ep3bs-1ep3bs](https://huggingface.co/Myaukko/videomae-base-finetuned-fighting-subset-5ep-5ep3bs-1ep3bs)
*   **Use Case**: Classifying a 2-3 second clip as "fighting" vs "non-fighting".

## 3. How to Use YOLOv8 Models in Python

### Prerequisites
```bash
pip install ultralytics
```

### Inference Script
```python
from ultralytics import YOLO
import cv2

# Load one of the downloaded models
model = YOLO('yolo_best.pt')

# Run inference on a video file
# 'conf' is the confidence threshold (0.25 is a good start)
results = model.predict(source='input_movie.mp4', conf=0.25, save=False)

# Iterate through frames
for result in results:
    # Get detection boxes
    boxes = result.boxes
    for box in boxes:
        # Get class ID and confidence
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Mapping depends on the specific model's data.yaml
        # For violence models, usually class 0 or 1 indicates violence
        print(f"Detected Class {cls_id} with {conf:.2f} confidence at frame {result.path}")

# Note: For automated filtering, you can record the timestamps where 
# detections occur and use them to trigger muting or skipping.
```

## 4. Summary Table

| Category | Model Name | Source | Strength |
| :--- | :--- | :--- | :--- |
| **Violence** | `siwon23/violence-detection-yolov8` | Hugging Face | General fighting/violence. |
| **Weapons** | `Hadi959/weapon-detection-yolov8` | Hugging Face | Specific object detection (guns/knives). |
| **Blood** | `Notacodinggeek/yolov8n-blood-violence` | Hugging Face | Graphic content/blood. |
| **Action** | `Myaukko/videomae-base...` | Hugging Face | Temporal context (better for long fights). |
