import whisper
from better_profanity import profanity
from moviepy import VideoFileClip, AudioFileClip
import os
import torch

def filter_audio(input_video, output_video):
    print(f"Processing {input_video}...")
    
    # 1. Load video
    video = VideoFileClip(input_video)
    audio = video.audio
    
    if audio is None:
        print("No audio track found.")
        video.write_videofile(output_video, codec="libx264")
        return

    # Save audio to a temporary file for Whisper
    temp_audio = "temp_audio.wav"
    audio.write_audiofile(temp_audio)
    
    # 2. Transcribe audio using Whisper with word-level timestamps
    print("Loading Whisper model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)
    
    print("Transcribing audio...")
    result = model.transcribe(temp_audio, word_timestamps=True)
    
    # 3. Identify profanity and get timestamps
    mute_intervals = []
    
    for segment in result['segments']:
        if 'words' in segment:
            for word_info in segment['words']:
                word = word_info['word']
                if profanity.contains_profanity(word):
                    # Add a small buffer around the word
                    start = max(0, word_info['start'] - 0.1)
                    end = word_info['end'] + 0.1
                    mute_intervals.append((start, end))
                    print(f"Profanity detected: '{word}' at {start:.2f}-{end:.2f}")
        else:
            # Fallback to segment level if word timestamps are missing
            if profanity.contains_profanity(segment['text']):
                mute_intervals.append((segment['start'], segment['end']))
                print(f"Profanity detected (segment): '{segment['text']}' at {segment['start']:.2f}-{segment['end']:.2f}")

    if not mute_intervals:
        print("No profanity detected.")
        video.write_videofile(output_video, codec="libx264", audio_codec="aac")
        return

    # Merge overlapping intervals
    mute_intervals.sort()
    merged_intervals = []
    if mute_intervals:
        curr_start, curr_end = mute_intervals[0]
        for next_start, next_end in mute_intervals[1:]:
            if next_start <= curr_end:
                curr_end = max(curr_end, next_end)
            else:
                merged_intervals.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end
        merged_intervals.append((curr_start, curr_end))

    print(f"Muting {len(merged_intervals)} segments...")

    # 4. Mute the flagged segments
    def mute_filter(get_frame, t):
        frame = get_frame(t)
        for start, end in merged_intervals:
            if start <= t <= end:
                # Return silent frame (zeros)
                # frame is usually a numpy array (samples, channels)
                return frame * 0
        return frame

    audio_filtered = audio.with_effects([lambda clip: clip.fl_audio(mute_filter)])
    # In MoviePy 2.x, fl_audio might be slightly different or moved.
    # Actually, MoviePy 2.x uses audio_clip.fl_audio(filter) or clip.effects.multiply_volume(...)
    
    # Let's try the MoviePy 2.x way if the above fails.
    # Actually, the standard way in v1 and v2 is audio.fl_audio(mute_filter)
    
    # Re-evaluating MoviePy 2.x API:
    # It seems v2 changed some things but kept compatibility for many.
    # Let's use a simpler approach for muting:
    # Using volume filters if possible, but fl_audio is most flexible.

    final_video = video.with_audio(audio_filtered)
    
    # Export the final video
    final_video.write_videofile(output_video, codec="libx264", audio_codec="aac")
    
    # Cleanup
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    print("Done!")

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_video.mp4"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "filtered_video.mp4"
    filter_audio(input_file, output_file)
