import cv2
import numpy as np
import whisper
import torch
import os
import sys
import argparse
from better_profanity import profanity
from nudenet import NudeDetector
from ultralytics import YOLO
from moviepy import VideoFileClip

class SafeSeeing:
    """
    Safe Seeing: An automated pipeline to detect and filter NSFW content, 
    violence, and profanity from videos.
    """
    def __init__(self, model_dir="models", sample_fps=2, confidence_threshold=0.3):
        self.sample_fps = sample_fps
        self.confidence_threshold = confidence_threshold
        self.model_dir = model_dir
        
        print("Initializing models...")
        # Audio - Whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.whisper_model = whisper.load_model("base", device=self.device)
        
        # Nudity - NudeNet
        self.nude_detector = NudeDetector()
        self.nsfw_labels = [
            "BUTTOCKS_EXPOSED", "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED",
            "MALE_BREAST_EXPOSED", "ANUS_EXPOSED", "MALE_GENITALIA_EXPOSED"
        ]
        
        # Violence/Weapons - YOLOv8
        self.violence_model = self._load_yolo('yolo_best.pt')
        self.weapon_model = self._load_yolo('weapon_yolov8.pt')
        self.blood_model = self._load_yolo('yolov8n-blood-violence.pt')
        
        # Internal Caches
        self.mute_intervals = []
        self.visual_detections = {} # t -> {'nsfw': [boxes], 'violence': bool}

    def _load_yolo(self, filename):
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            # Fallback to current directory for backwards compatibility
            if os.path.exists(filename):
                path = filename
            else:
                raise FileNotFoundError(f"Model file not found: {path}. Run setup_models.py first.")
        return YOLO(path)

    def pre_scan(self, video_path):
        """
        Scan the video for offensive content and cache timestamps.
        """
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # 1. Audio Scan
        if video.audio:
            print("Pre-scanning audio for profanity...")
            temp_audio = "temp_audio_scan.wav"
            video.audio.write_audiofile(temp_audio, logger=None)
            audio_result = self.whisper_model.transcribe(temp_audio, word_timestamps=True)
            
            self.mute_intervals = []
            for segment in audio_result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        if profanity.contains_profanity(word_info['word']):
                            start = max(0, word_info['start'] - 0.1)
                            end = word_info['end'] + 0.1
                            self.mute_intervals.append((start, end))
                elif profanity.contains_profanity(segment['text']):
                    self.mute_intervals.append((segment['start'], segment['end']))
            
            # Merge intervals
            self.mute_intervals.sort()
            merged = []
            if self.mute_intervals:
                curr_s, curr_e = self.mute_intervals[0]
                for n_s, n_e in self.mute_intervals[1:]:
                    if n_s <= curr_e:
                        curr_e = max(curr_e, n_e)
                    else:
                        merged.append((curr_s, curr_e))
                        curr_s, curr_e = n_s, n_e
                merged.append((curr_s, curr_e))
            self.mute_intervals = merged
            
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        else:
            print("No audio track found. Skipping audio scan.")
        
        # 2. Visual Scan
        print(f"Pre-scanning visual content at {self.sample_fps} fps...")
        for t in np.arange(0, duration, 1.0 / self.sample_fps):
            frame = video.get_frame(t)
            
            # Nudity Detection
            nude_detections = self.nude_detector.detect(frame)
            nsfw_boxes = [d for d in nude_detections if d['class'] in self.nsfw_labels and d['score'] > self.confidence_threshold]
            
            # Violence/Weapon Detection
            has_violence = False
            for model in [self.violence_model, self.weapon_model, self.blood_model]:
                results = model.predict(frame, conf=self.confidence_threshold, verbose=False)
                if any(len(r.boxes) > 0 for r in results):
                    has_violence = True
                    break
            
            self.visual_detections[t] = {
                'nsfw': nsfw_boxes,
                'violence': has_violence
            }
            
            if nsfw_boxes or has_violence:
                flags = []
                if nsfw_boxes: flags.append("NSFW")
                if has_violence: flags.append("Violence/Weapon")
                print(f"Content flagged at {t:.2f}s: {', '.join(flags)}")

        video.close()

    def apply_filters(self, input_path, output_path):
        """
        Apply the cached filters and write the output video.
        """
        video = VideoFileClip(input_path)
        
        def process_frame(get_frame, t):
            frame = get_frame(t)
            
            # Find closest cached detection
            sample_times = sorted(self.visual_detections.keys())
            if not sample_times:
                return frame
                
            closest_t = min(sample_times, key=lambda x: abs(x - t))
            if abs(closest_t - t) > (1.0 / self.sample_fps):
                return frame
                
            detections = self.visual_detections[closest_t]
            
            # Apply Violence Blur (Full Frame)
            if detections['violence']:
                return cv2.GaussianBlur(frame, (99, 99), 0)
            
            # Apply Nudity Blur (Targeted)
            for d in detections['nsfw']:
                x, y, w, h = d['box']
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(frame.shape[1], int(x + w)), min(frame.shape[0], int(y + h))
                if x2 > x1 and y2 > y1:
                    region = frame[y1:y2, x1:x2]
                    if region.size > 0:
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (51, 51), 0)
            
            return frame

        def process_audio(get_frame, t):
            frame = get_frame(t)
            for start, end in self.mute_intervals:
                if start <= t <= end:
                    return frame * 0
            return frame

        print("Applying filters and rendering final video...")
        # Apply visual filter
        processed_video = video.transform(process_frame)
        
        # Apply audio filter
        if video.audio:
            filtered_audio = video.audio.with_effects([lambda clip: clip.fl_audio(process_audio)])
            processed_video = processed_video.with_audio(filtered_audio)
        
        processed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        video.close()

    def run(self, input_path, output_path):
        try:
            self.pre_scan(input_path)
            self.apply_filters(input_path, output_path)
            print(f"\nSuccess! Safe video saved to: {output_path}")
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Safe Seeing: Automated content filtering for videos.")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output", help="Path to output video file")
    parser.add_argument("--model_dir", default="models", help="Directory containing model weights")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling rate for detection (default: 2.0)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    safe_seeing = SafeSeeing(model_dir=args.model_dir, sample_fps=args.fps, confidence_threshold=args.conf)
    safe_seeing.run(args.input, args.output)

if __name__ == "__main__":
    main()
