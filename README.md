# Smart Object Tracking System

A real-time object detection and tracking system for edge devices with online/offline capabilities and OTA model updates.

## Features

- **Real-time object detection** using YOLOv5 models
- **Multiple tracking algorithms** (DeepSORT and ByteTrack)
- **Online and offline modes** with automatic switching based on connectivity
- **OTA (Over-The-Air) model updates** for keeping models current
- **Offline queue** for storing detections when operating without connectivity
- **Configurable settings** via YAML configuration file
- **Display and video saving** capabilities
- **Automatic model downloads** from Ultralytics repositories

## System Requirements

- Python 3.9+
- OpenCV 4.5+
- PyTorch 2.0+
- CUDA-compatible GPU (optional, improves performance)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart_object_tracking.git
   cd smart_object_tracking
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv_py39
   # On Windows
   venv_py39\Scripts\activate
   # On Linux/Mac
   source venv_py39/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install torch torchvision torchaudio
   pip install opencv-python
   pip install pyyaml
   pip install ultralytics
   ```

   Alternatively, you can use the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the project structure:
   ```bash
   mkdir -p configs
   mkdir -p models
   mkdir -p logs
   ```

5. Create the configuration file in `configs/settings.yaml` using the template provided in this repository.

6. Download a sample video for testing (optional):
   ```bash
   # Create a directory for test videos
   mkdir -p test_videos
   
   # Download a sample video
   wget -P test_videos https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4
   ```

## Project Structure

```
smart_object_tracking/
├── main.py                  # Main application entry point
├── detect_and_track.py      # Detection and tracking implementation
├── configs/
│   └── settings.yaml        # Configuration file
├── tracker/
│   ├── __init__.py          # Tracker package initialization
│   └── tracker.py           # Object tracker implementation
├── updater/
│   ├── __init__.py          # Updater package initialization
│   └── ota_updater.py       # OTA updater implementation
├── utils/
│   ├── __init__.py          # Utils package initialization
│   ├── connectivity.py      # Network connectivity checker
│   ├── dataloaders.py       # Data loading utilities (YOLOv5 compatibility)
│   └── try_except.py        # Error handling utilities (YOLOv5 compatibility)
├── offline_queue.py         # Offline queue implementation
├── models/                  # Model storage directory
├── logs/                    # Log files directory
├── test_videos/             # Sample videos for testing
└── requirements.txt         # Package dependencies
```

## Required Files for YOLOv5 Compatibility

The project includes special compatibility files for working with YOLOv5:

1. **utils/try_except.py**:
   ```python
   class TryExcept:
       def __init__(self, func):
           self.func = func
           
       def __call__(self, *args, **kwargs):
           try:
               return self.func(*args, **kwargs)
           except Exception as e:
               print(f'Error in {self.func.__name__}: {e}')
               return None
   ```

2. **utils/dataloaders.py**:
   This file provides the necessary functions for data loading compatibility with YOLOv5, including `exif_transpose`, `letterbox`, and `img2label_paths`.

3. **utils/__init__.py**:
   ```python
   from .try_except import TryExcept
   from .dataloaders import exif_transpose, letterbox, img2label_paths
   
   __all__ = [
       'TryExcept',
       'exif_transpose', 
       'letterbox',
       'img2label_paths'
   ]
   ```

## Configuration

The system is configured via the `configs/settings.yaml` file. The main configuration sections include:

- **System information** - Version and description
- **Model settings** - Model paths for online and offline modes
- **Input settings** - Camera and resolution configurations
- **Detection settings** - Confidence thresholds and detection frequency
- **Tracking settings** - Tracker type and parameters
- **Offline mode settings** - Queue configuration
- **Update settings** - OTA update configuration
- **Logging settings** - Log level and output options
- **Device settings** - CPU/CUDA device selection

Example configuration:

```yaml
# System information
system:
  name: "Smart Object Tracking System"
  version: "1.0.0"
  description: "Real-time object detection and tracking system for edge devices"

# Model settings
models:
  directory: "models"
  online_model: "yolov5s.pt"
  offline_model: "yolov5s-fp16.onnx"

# Detection settings
detection:
  confidence: 0.25
  iou_threshold: 0.45
  frequency: 5

# Device settings
device: "cpu"  # or "cuda"
```

## Usage

### Basic Usage

To start the system with default settings:

```bash
python main.py
```

### Command-line Options

- `--source` - Specify input source (0 for webcam, path for video file)
- `--display` - Display output in a window
- `--save-video` - Save output video
- `--output-dir` - Directory to save output files (default: 'output')
- `--tracker` - Specify tracker type ('deep_sort' or 'byte_track')
- `--config` - Path to custom config file

### Examples

1. Run with webcam and display output:
   ```bash
   python main.py --display
   ```

2. Process a video file:
   ```bash
   python main.py --source test_videos/person-bicycle-car-detection.mp4 --display
   ```

3. Use a specific tracker:
   ```bash
   python main.py --display --tracker byte_track
   ```

4. Save output video:
   ```bash
   python main.py --display --save-video --output-dir my_results
   ```

## Camera Access in Different Environments

### Native Operating System

On a native operating system (Windows, macOS, Linux), camera access typically works out of the box:

```bash
python main.py --display
```

### Windows Subsystem for Linux (WSL)

Camera access in WSL is challenging but possible:

#### Option 1: Use a video file instead
```bash
python main.py --source test_videos/person-bicycle-car-detection.mp4 --display
```

#### Option 2: Configure WSL2 for USB camera access

1. Identify your camera in PowerShell (Admin):
   ```powershell
   usbipd wsl list
   ```

2. Attach the camera to WSL:
   ```powershell
   usbipd wsl attach --busid <your-camera-busid>
   ```

3. In WSL, verify the camera is recognized:
   ```bash
   ls /dev/video*
   ```

4. Run the tracking system:
   ```bash
   python main.py --display
   ```

## Troubleshooting

### Common Issues and Solutions

#### Numpy Array vs PyTorch Tensor Error

```
TypeError: conv2d() received an invalid combination of arguments - got (numpy.ndarray, Parameter, NoneType, tuple, tuple, tuple, int)
```

**Solution**:
- This error occurs because the model expects PyTorch tensors but is receiving numpy arrays
- The preprocess function in `detect_and_track.py` handles the conversion from numpy arrays to PyTorch tensors
- Make sure you're using the latest version that properly converts the input data types
- The conversion includes:
  - Converting from BGR to RGB format (OpenCV uses BGR, PyTorch expects RGB)
  - Converting from numpy array to PyTorch tensor
  - Changing from HWC to CHW format (Height-Width-Channel to Channel-Height-Width)
  - Normalizing values from 0-255 to 0-1
  - Adding batch dimension

#### Import Errors with YOLOv5

```
ImportError: cannot import name 'TryExcept' from 'utils'
```

**Solution**: 
- Make sure you have the `utils/try_except.py` file with the `TryExcept` class implementation.
- Verify that `utils/__init__.py` correctly exports the `TryExcept` class.

#### Missing Dataloaders Module

```
ModuleNotFoundError: No module named 'utils.dataloaders'
```

**Solution**:
- Create the `utils/dataloaders.py` file with the required functions.
- Ensure `utils/__init__.py` correctly imports and exports these functions.

#### Camera Access Issues

```
VIDEOIO(V4L2:/dev/video0): can't open camera by index
```

**Solution**:
- Use a video file instead of a camera for testing:
  ```bash
  python main.py --source test_videos/person-bicycle-car-detection.mp4 --display
  ```
- If using WSL, follow the WSL camera access instructions in the previous section.
- Check if your camera is being used by another application.
- Try a different camera index (e.g., `--source 1`).

#### Type Errors with Source

```
AttributeError: 'int' object has no attribute 'isdigit'
```

**Solution**:
- Make sure you're using the latest version of `detect_and_track.py` which handles both integer and string source types.

#### CUDA Out-of-Memory

```
CUDA out of memory
```

**Solution**:
- Switch to CPU mode in the config file: `device: "cpu"`
- Reduce input resolution in the `input` section of the config.
- Reduce batch size if applicable.

## OTA Updates

The system can automatically check for and download model updates. To configure this:

1. Set up the update server endpoints in `settings.yaml`:
   ```yaml
   updates:
     enabled: true
     check_url: "https://api.example.com/model-updates/check"
     download_url: "https://storage.example.com/models/"
     interval: 86400  # Check every 24 hours
   ```

2. The system will automatically check for updates when online.

## Advanced Features

### Running in Headless Mode

For systems without a display (like servers or IoT devices), you can run without the `--display` flag and save results:

```bash
python main.py --source test_videos/input.mp4 --save-video --output-dir results
```

### Using with IP Cameras

You can use RTSP, HTTP, or other streaming protocols:

```bash
python main.py --source rtsp://username:password@192.168.1.100:554/stream
```

### Running Multiple Instances

For monitoring multiple cameras:

```bash
# Terminal 1
python main.py --source 0 --display --output-dir camera1

# Terminal 2
python main.py --source 1 --display --output-dir camera2
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [DeepSORT](https://github.com/nwojke/deep_sort) tracking algorithm
- [ByteTrack](https://github.com/ifzhang/ByteTrack) tracking algorithm