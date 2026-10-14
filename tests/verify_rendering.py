
import cv2
import numpy as np
import os
import sys
from unittest.mock import MagicMock

# Mock heavy AI libraries before importing SafeSeeing
sys.modules['whisper'] = MagicMock()
sys.modules['nudenet'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['torch'] = MagicMock()

from moviepy import VideoFileClip, ColorClip, AudioArrayClip, CompositeVideoClip
from safe_seeing import SafeSeeing

class VerifySafeSeeing(SafeSeeing):
    def __init__(self, sample_fps=2):
        self.sample_fps = sample_fps
        self.mute_intervals = []
        self.visual_detections = {}
        self.confidence_threshold = 0.3

    def inject_detections(self, mute_intervals, visual_detections):
        self.mute_intervals = mute_intervals
        self.visual_detections = visual_detections

def create_synthetic_video(path, duration=10):
    print(f"Creating synthetic video: {path}")
    # Create a red video
    clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=duration)
    
    # Create a simple beep audio (440Hz sine wave)
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
    # Convert to 2D array for stereo if needed, but mono is fine
    audio_clip = AudioArrayClip(audio_data, fps=sample_rate)
    
    clip = clip.with_audio(audio_clip)
    clip.write_videofile(path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    clip.close()

def verify_output(path):
    print(f"Verifying output: {path}")
    video = VideoFileClip(path)
    
    # Check frame at 1.5s (Profanity - should be muted)
    # We can't easily check audio "mutedness" with moviepy without extracting it,
    # but we can check visual blurs.
    
    # Check frame at 3s (Violence - should be full frame blurred)
    frame_3s = video.get_frame(3.0)
    # A solid red frame (255, 0, 0) should still be red after Gaussian blur, 
    # but let's see if we can detect the blur.
    # Actually, Gaussian blur on a perfectly uniform frame does nothing.
    # Let's add a white square to the synthetic video so we can detect blur.
    
    video.close()
    return True

if __name__ == "__main__":
    input_path = "verify_input.mp4"
    output_path = "verify_output.mp4"
    
    # Improved synthetic video with a pattern to detect blur
    print("Creating synthetic video with pattern...")
    duration = 10
    # Red background
    bg = ColorClip(size=(640, 480), color=(255, 0, 0), duration=duration)
    # Smaller white square in the middle (20x20)
    square = ColorClip(size=(20, 20), color=(255, 255, 255), duration=duration).with_position('center')
    clip = CompositeVideoClip([bg, square])
    
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
    audio_clip = AudioArrayClip(audio_data, fps=sample_rate)
    clip = clip.with_audio(audio_clip)
    clip.write_videofile(input_path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    
    verifier = VerifySafeSeeing(sample_fps=2)
    
    # Inject detections
    # 1. Mute at 1-2s
    mute_intervals = [(1.0, 2.0)]
    
    # 2. Visual Detections
    visual_detections = {
        # 3s: Violence (Full Blur)
        3.0: {'nsfw': [], 'violence': True},
        # 7s: NSFW (Targeted Blur at center where the square is)
        7.0: {'nsfw': [{'box': [270, 190, 100, 100], 'class': 'FEMALE_BREAST_EXPOSED', 'score': 0.9}], 'violence': False}
    }
    verifier.inject_detections(mute_intervals, visual_detections)
    
    print("Running apply_filters...")
    verifier.apply_filters(input_path, output_path)
    
    print("Programmatic verification...")
    out_video = VideoFileClip(output_path)
    
    # Frame at 0s (Original)
    f0 = out_video.get_frame(0.1)
    # Center pixel should be white (255, 255, 255)
    center_val = f0[240, 320]
    print(f"Frame 0.1s center pixel: {center_val}")
    
    # Frame at 3s (Violence - Full Blur)
    f3 = out_video.get_frame(3.0)
    center_val_3 = f3[240, 320]
    corner_val_3 = f3[10, 10]
    print(f"Frame 3.0s center pixel: {center_val_3}")
    print(f"Frame 3.0s corner pixel: {corner_val_3}")
    
    # Frame at 7s (NSFW - Targeted Blur)
    f7 = out_video.get_frame(7.0)
    center_val_7 = f7[240, 320]
    print(f"Frame 7.0s center pixel: {center_val_7}")
    
    # Check audio muting
    print("Verifying audio muting...")
    # Get 0.1s of audio around 0.5s (should NOT be muted)
    audio_05 = out_video.audio.subclipped(0.45, 0.55).to_soundarray(fps=1000)
    # Get 0.1s of audio around 1.5s (should BE muted)
    audio_15 = out_video.audio.subclipped(1.45, 1.55).to_soundarray(fps=1000)
    
    max_05 = np.abs(audio_05).max()
    max_15 = np.abs(audio_15).max()
    print(f"Max audio amplitude at 0.5s: {max_05}")
    print(f"Max audio amplitude at 1.5s: {max_15}")
    
    is_muted_15 = max_15 < 1e-5
    is_not_muted_05 = max_05 > 0.1
    
    if is_muted_15 and is_not_muted_05:
        print("SUCCESS: Audio muting correctly applied!")
    else:
        print("FAILURE: Audio muting not applied correctly.")
        print(f"Muted at 1.5s: {is_muted_15}")
        print(f"Not muted at 0.5s: {is_not_muted_05}")

    # Check if center is whitish at 0.1s and blurred at 3.0s and 7.0s
    is_whitish_0 = np.all(center_val > 240)
    is_not_whitish_3 = np.any(center_val_3 < 100) # Should have some red background bleed
    is_not_whitish_7 = np.any(center_val_7 < 200) # Targeted blur
    
    if is_whitish_0 and is_not_whitish_3 and is_not_whitish_7:
        print("SUCCESS: Rendering path correctly applied filters!")
    else:
        print("FAILURE: Rendering path did not apply filters as expected.")
        print(f"Whitish at 0.1s: {is_whitish_0}")
        print(f"Blurred at 3.0s: {is_not_whitish_3}")
        print(f"Blurred at 7.0s: {is_not_whitish_7}")
        # Note: If GaussianBlur on uniform regions, it might still look white. 
        # But the edges of the square should definitely be blurred.
    
    out_video.close()
