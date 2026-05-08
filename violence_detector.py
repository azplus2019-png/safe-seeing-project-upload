import cv2
import numpy as np
from ultralytics import YOLO
from moviepy import VideoFileClip
import os

class ViolenceWeaponFilter:
    def __init__(self, confidence_threshold=0.3, sample_fps=2):
        # Load models
        print("Loading YOLOv8 models...")
        self.violence_model = YOLO('yolo_best.pt')
        self.weapon_model = YOLO('weapon_yolov8.pt')
        self.blood_model = YOLO('yolov8n-blood-violence.pt')
        
        self.confidence_threshold = confidence_threshold
        self.sample_fps = sample_fps
        self.detections_cache = {} # timestamp -> True/False (should blur)

    def is_nsfw_frame(self, frame):
        # Run inference with all models
        # Note: We just need to know if the frame contains violence/weapons/blood
        
        # 1. Violence check
        results_v = self.violence_model.predict(frame, conf=self.confidence_threshold, verbose=False)
        if any(len(r.boxes) > 0 for r in results_v):
            return True
            
        # 2. Weapon check
        results_w = self.weapon_model.predict(frame, conf=self.confidence_threshold, verbose=False)
        if any(len(r.boxes) > 0 for r in results_w):
            return True
            
        # 3. Blood check
        results_b = self.blood_model.predict(frame, conf=self.confidence_threshold, verbose=False)
        if any(len(r.boxes) > 0 for r in results_b):
            return True
            
        return False

    def get_should_blur_for_timestamp(self, t):
        sample_times = sorted(self.detections_cache.keys())
        if not sample_times:
            return False
        
        closest_t = min(sample_times, key=lambda x: abs(x - t))
        if abs(closest_t - t) > (1.0 / self.sample_fps):
            return False
            
        return self.detections_cache[closest_t]

    def process_frame(self, get_frame, t):
        frame = get_frame(t)
        if self.get_should_blur_for_timestamp(t):
            # Apply full frame blur or black out
            # Blurring the whole frame for violence/weapons as requested
            return cv2.GaussianBlur(frame, (99, 99), 0)
        return frame

def filter_video_violence(input_path, output_path):
    v_filter = ViolenceWeaponFilter(sample_fps=2)
    
    video = VideoFileClip(input_path)
    
    print(f"Pre-scanning video for violence/weapons at {v_filter.sample_fps} fps...")
    duration = video.duration
    for t in np.arange(0, duration, 1.0 / v_filter.sample_fps):
        frame = video.get_frame(t)
        should_blur = v_filter.is_nsfw_frame(frame)
        v_filter.detections_cache[t] = should_blur
        if should_blur:
            print(f"Violence/Weapon detected at {t:.2f}s")

    print("Applying filter and writing output...")
    processed_video = video.transform(v_filter.process_frame)
    
    processed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python violence_detector.py input_video output_video")
    else:
        filter_video_violence(sys.argv[1], sys.argv[2])
