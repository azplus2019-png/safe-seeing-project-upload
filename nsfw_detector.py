import cv2
import numpy as np
from nudenet import NudeDetector
from moviepy import VideoFileClip
import os

class VisualNSFWFilter:
    def __init__(self, confidence_threshold=0.3, sample_fps=2):
        self.detector = NudeDetector()
        self.confidence_threshold = confidence_threshold
        self.sample_fps = sample_fps
        self.nsfw_labels = [
            "BUTTOCKS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED",
            "ANUS_EXPOSED",
            "MALE_GENITALIA_EXPOSED"
        ]
        self.detections_cache = {}

    def get_detections_for_timestamp(self, t):
        # Find the closest sampled timestamp
        sample_times = sorted(self.detections_cache.keys())
        if not sample_times:
            return []
        
        # Simple approach: find the closest sample
        closest_t = min(sample_times, key=lambda x: abs(x - t))
        
        # If the closest sample is too far away (e.g. > 1/sample_fps), maybe return nothing
        if abs(closest_t - t) > (1.0 / self.sample_fps):
            return []
            
        return self.detections_cache[closest_t]

    def blur_regions(self, image, detections):
        for detection in detections:
            if detection['class'] in self.nsfw_labels and detection['score'] > self.confidence_threshold:
                box = detection['box']
                x, y, w, h = box
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
                
                if x2 > x1 and y2 > y1:
                    region = image[y1:y2, x1:x2]
                    # Check if region is valid
                    if region.size > 0:
                        blurred_region = cv2.GaussianBlur(region, (51, 51), 0)
                        image[y1:y2, x1:x2] = blurred_region
        return image

    def process_frame(self, get_frame, t):
        frame = get_frame(t)
        detections = self.get_detections_for_timestamp(t)
        return self.blur_regions(frame, detections)

def filter_video_nsfw(input_path, output_path):
    nsfw_filter = VisualNSFWFilter(sample_fps=2)
    
    video = VideoFileClip(input_path)
    
    print(f"Pre-scanning video at {nsfw_filter.sample_fps} fps...")
    duration = video.duration
    for t in np.arange(0, duration, 1.0 / nsfw_filter.sample_fps):
        frame = video.get_frame(t)
        detections = nsfw_filter.detector.detect(frame)
        nsfw_filter.detections_cache[t] = detections
        if any(d['class'] in nsfw_filter.nsfw_labels for d in detections):
            print(f"NSFW detected at {t:.2f}s")

    print("Applying blur and writing output...")
    # fl_image doesn't give timestamp easily, use fl
    processed_video = video.transform(nsfw_filter.process_frame, apply_to=['mask'])
    # In MoviePy 2.x, fl is often replaced by transform or similar.
    # Actually VideoFileClip.transform(func) where func(get_frame, t)
    
    processed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python nsfw_detector.py input_video output_video")
    else:
        filter_video_nsfw(sys.argv[1], sys.argv[2])
