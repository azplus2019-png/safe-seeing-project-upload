# Safe Seeing 🛡️📺

Safe Seeing is an automated AI-powered video filtering tool designed to make content safer for children. It identifies and obscures profanity, nudity, and violence in video files.

## Features

- **Audio Profanity Filtering**: Automatically silences offensive language using AI-powered speech-to-text.
- **Visual NSFW Detection**: Blurs exposed nudity and adult content.
- **Violence & Weapon Detection**: Identifies fighting, weapons (guns/knives), and blood, applying a safety blur to these scenes.
- **Efficient Processing**: Modular pipeline that pre-scans content and applies filters in a single high-performance pass.

## AI Stack

Safe Seeing leverages state-of-the-art AI models:

- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper) for high-accuracy audio transcription with word-level timestamps.
- **Visual Content Safety**: [NudeNet](https://github.com/notAI-tech/NudeNet) for detecting and localizing adult content.
- **Object Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) with specialized fine-tuned models for violence and weapon detection.
- **Video Processing**: [MoviePy](https://github.com/Zulko/moviepy) and [OpenCV](https://github.com/opencv/opencv) for robust video manipulation and rendering.

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed on your system

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/safe-seeing.git
   cd safe-seeing
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights**:
   The project requires several pre-trained models. We provide a script to download them automatically:
   ```bash
   python setup_models.py
   ```

## Usage

Run the integrated pipeline using the main script:

```bash
python safe_seeing.py input_movie.mp4 output_safe_movie.mp4
```

### Options

- `--model_dir`: Directory where models are stored (default: `models/`)
- `--fps`: Sampling rate for scanning (default: `2.0`). Higher values increase accuracy but take longer.
- `--conf`: Confidence threshold for detections (default: `0.3`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the developers of Whisper, NudeNet, and Ultralytics for providing the foundational models for this project.
