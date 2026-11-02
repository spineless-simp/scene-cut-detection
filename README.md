# Scene Cut Detector

A Python CLI tool for detecting hard scene cuts in video files using FFmpeg streaming. Processes videos with constant memory usage regardless of file size.

## Features

- **Streaming processing**: No intermediate files or full video loading into memory
- **FFmpeg integration**: Uses FFmpeg's hardware-accelerated decoding
- **Configurable detection**: Adjust threshold and minimum separation between cuts
- **Progress reporting**: Real-time feedback on frames processed, FPS, and time remaining
- **Multiple output formats**: CSV and JSON output with accurate timestamps
- **Downscaling**: Process at lower resolution for faster analysis
- **Cross-platform**: Works on Linux, macOS, and Windows

## Requirements

- Python 3.11+
- FFmpeg 4.0+ (with ffprobe)
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/scene-cut-detector.git
cd scene-cut-detector

# Install dependencies
pip install -e .
```

Or install directly from PyPI (when available):

```bash
pip install scene-cut-detector
```

## Usage

### Basic Usage

```bash
# Detect cuts with default settings, output CSV to stdout
scene-cut-detector video.mp4

# Custom threshold and output file
scene-cut-detector video.mp4 -t 0.5 -o cuts.csv

# JSON output with pretty printing
scene-cut-detector video.mp4 -j --pretty

# Minimum 1 second between cuts
scene-cut-detector video.mp4 --min-sep 1.0

# Process at higher resolution for more accuracy
scene-cut-detector video.mp4 -w 640 -H 360
```

### Command-Line Options

```
scene-cut-detector [OPTIONS] INPUT

Arguments:
  INPUT                   Input video file path (required)

Options:
  -o, --output PATH       Output file path (CSV or JSON based on extension)
  -t, --threshold FLOAT   Difference threshold for cut detection (0-1, default: 0.3)
  --min-sep FLOAT         Minimum seconds between cuts (default: 0.5)
  -w, --width INTEGER     Analysis frame width (default: 160)
  -H, --height INTEGER    Analysis frame height (default: 90)
  -j, --json              Output in JSON format (overrides extension detection)
  -p, --pretty            Pretty-print JSON output
  --mean-diff             Use mean difference instead of max difference
  -v, --verbose           Enable verbose progress output
  --version               Show version and exit
  -h, --help              Show help message and exit
```

### Output Format

The tool outputs results in CSV or JSON format.

**CSV columns:**
- `cut_index`: Sequential cut number (0-based)
- `timestamp`: Timestamp in seconds
- `frame_index`: Frame index where cut occurred
- `difference`: Frame difference value that triggered the cut
- `threshold`: Threshold used for detection

**JSON structure:**
```json
{
  "source_file": "video.mp4",
  "total_duration": 123.45,
  "cuts": [
    {
      "cut_index": 0,
      "timestamp": 15.67,
      "frame_index": 470,
      "difference": 0.85,
      "threshold": 0.3
    }
  ]
}
```

## How It Works

1. **FFmpeg Streaming**: The tool uses FFmpeg to decode video frames and stream them via stdout without creating intermediate files.

2. **Grayscale Conversion**: Frames are converted to grayscale for faster comparison.

3. **Frame Difference**: Compares each frame with the previous one using pixel difference.

4. **Cut Detection**: When the difference exceeds the configured threshold, a scene cut is detected.

5. **Minimum Separation**: Ensures cuts are separated by at least the specified time to avoid duplicate detections.

## Example Output

```bash
$ scene-cut-detector sample.mp4 -v
Video info: 1920x1080
Duration: 00:02:03.45
FPS: 30.00
Processing at 160x90 for analysis

Processed 3705 frames (100%): 892.3 FPS, 0.0s remaining

Processed 3705 frames in 4.15s (892.3 FPS)
Detected 12 scene cuts
cut_index,timestamp,frame_index,difference,threshold
0,00:00:02.50,75,0.72,0.3
1,00:00:15.30,459,0.85,0.3
2,00:00:45.20,1356,0.91,0.3
...
```

## API

The tool is also available as a Python library:

```python
from scene_cut_detector import VideoDecoder, SceneCutDetector

# Decode video
decoder = VideoDecoder("video.mp4", analysis_width=160)
decoder.start()

# Detect cuts
detector = SceneCutDetector(threshold=0.3)
cuts = []

for frame_index, timestamp, frame_data in decoder.decode_frames():
    new_cuts = detector.process_frame(frame_index, timestamp, frame_data, 160, 90)
    cuts.extend(new_cuts)

decoder.stop()
```

## Development

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy scene_cut_detector/

# Format code
black scene_cut_detector/ tests/
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with FFmpeg for efficient video decoding
- Uses NumPy for fast array operations
