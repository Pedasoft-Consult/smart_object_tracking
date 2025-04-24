# Smart Object Tracking System

A real-time object detection and tracking system designed for edge devices with intermittent network connectivity.

## Features

- Real-time object detection using YOLOv5 and other models
- Multiple tracking algorithms (DeepSORT, ByteTrack)
- Online and offline operation modes with seamless switching
- Offline queue for storing detections when disconnected
- Over-the-air model updates
- REST API for remote monitoring and control
- Comprehensive model registry with support for multiple frameworks

## Architecture

![System Architecture](docs/system_architecture.png)

The system is composed of the following key components:

1. **Core Detection and Tracking**
   - Object detection using YOLOv5 (and other supported models)
   - Object tracking with DeepSORT or ByteTrack algorithms
   - Frame processing pipeline with configurable settings

2. **Connectivity Management**
   - Auto-switching between online/offline modes
   - Network status monitoring
   - Resource-efficient operation

3. **Offline Queue**
   - Priority-based storage of detection events
   - Compressed data storage
   - Batched uploading when connectivity is restored

4. **Model Registry**
   - Support for multiple model frameworks:
     - PyTorch
     - ONNX
     - TensorFlow / TensorFlow Lite
     - TensorRT
     - OpenVINO
   - Versioned model management
   - Automatic model selection based on device capabilities

5. **API and Monitoring**
   - REST API for remote control
   - Live video streaming
   - Status reporting and statistics
   - Configuration management

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV 4.5+
- PyTorch 1.9+ (for PyTorch models)
- ONNX Runtime (for ONNX models)
- TensorFlow 2.x (for TensorFlow models)
- Flask (for API server)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-object-tracking.git
cd smart-object-tracking

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build the Docker image
docker build -t smart-object-tracking .

# Run the container
docker run -p 5000:5000 --device=/dev/video0:/dev/video0 smart-object-tracking
```

## Configuration

Configuration is stored in `configs/settings.yaml`. Key settings include:

```yaml
# Models
models:
  directory: "models"
  online_model: "yolov5s.pt"
  offline_model: "yolov5s-fp16.onnx"

# Detection settings
detection:
  confidence: 0.25
  iou_threshold: 0.45
  frequency: 5  # Run detection every N frames

# Tracking settings
tracking:
  tracker: "deep_sort"  # Options: "deep_sort" or "byte_track"
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

# Offline mode
offline:
  queue_directory: "queue"
  max_queue_size: 1000
  sync_interval: 3600  # Sync every hour when going online
```

## Usage

### Basic Usage

Run the tracking system with default settings:

```bash
python main.py
```

### Command Line Options

```bash
python main.py --source 0  # Use camera 0
python main.py --source video.mp4  # Use video file
python main.py --display  # Show visual output
python main.py --save-video  # Save output video
python main.py --tracker byte_track  # Use ByteTrack algorithm
```

### Running the API Server

```bash
python api.py --host 0.0.0.0 --port 5000
```

The API server provides a web interface at http://localhost:5000 and REST endpoints at http://localhost:5000/api/*.

### Using the Model Registry

Register and use a new model:

```python
from model_registry import ModelRegistry, ModelType

# Initialize registry
registry = ModelRegistry(config)

# Register a new model
model_id = registry.register_model(
    model_path="models/custom_model.onnx",
    name="Custom ONNX Model",
    description="My custom object detection model",
    tags=["custom", "optimized"],
    is_default=True  # Set as default model
)

# Load the model for inference
model, preprocess = registry.load_model(model_id)
```

### Using the Enhanced Offline Queue

```python
from enhanced_offline_queue import EnhancedOfflineQueue

# Initialize queue
queue = EnhancedOfflineQueue("queue_dir")

# Add detection with priority
queue.add(
    frame_id=123,
    detections=[{"bbox": [100, 200, 150, 250], "class_id": 0, "confidence": 0.95}],
    image=frame,  # Optional frame image
    priority=50   # Lower number = higher priority
)

# Add critical detection (highest priority)
queue.add_critical_priority(
    frame_id=124,
    detections=[{"bbox": [100, 200, 150, 250], "class_id": 0, "confidence": 0.98}],
    image=frame
)

# Upload pending items when online
queue.upload_pending_items(url="https://api.example.com/upload")
```

### Using the API Client

```python
from api_client import TrackingAPIClient, TrackingAPIMonitor

# Create client
client = TrackingAPIClient("http://localhost:5000")

# Get system status
status = client.get_status()
print(f"System status: {status}")

# Start tracking
client.start_tracking(source=0)  # Use default camera

# Monitor the system
monitor = TrackingAPIMonitor(client)

def on_status_update(status):
    print(f"FPS: {status.get('fps', 0):.1f}, Tracks: {status.get('tracks', 0)}")

def on_frame_update(frame):
    # Process the frame (e.g., display, save, analyze)
    pass

monitor.add_status_callback(on_status_update)
monitor.add_frame_callback(on_frame_update)
monitor.start()

# ... later
client.stop_tracking()
monitor.stop()
```

## Testing

Run the test suite:

```bash
cd tests
pytest
```

Run performance benchmarks:

```bash
pytest test_performance.py -v
```

## API Reference

The REST API provides the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get current system status |
| `/api/config` | GET | Get current configuration |
| `/api/config` | POST | Update configuration |
| `/api/start` | POST | Start tracking |
| `/api/stop` | POST | Stop tracking |
| `/api/frame` | GET | Get latest frame as JPEG |
| `/api/stream` | GET | Stream video frames |
| `/api/queue/stats` | GET | Get offline queue statistics |
| `/api/queue/sync` | POST | Trigger manual queue sync |
| `/api/models` | GET | List available models |
| `/api/models/update` | POST | Check for model updates |
| `/api/update_history` | GET | Get model update history |

## Project Structure

```
smart_object_tracking/
├── api.py                  # API server implementation
├── api_client.py           # API client library
├── configs/                # Configuration files
│   └── settings.yaml       # Main configuration file
├── detect_and_track.py     # Core detection and tracking
├── enhanced_offline_queue.py # Enhanced offline queue
├── main.py                 # Main entry point
├── model_registry.py       # Model management system
├── models/                 # Model storage
├── offline_queue.py        # Basic offline queue (legacy)
├── tests/                  # Test suite
│   ├── conftest.py         # Test fixtures
│   ├── test_model_registry.py
│   ├── test_offline_queue.py
│   ├── test_ota_updater.py
│   ├── test_performance.py
│   ├── test_tracker.py
│   └── test_utils.py
├── tracker/                # Tracking implementations
│   ├── tracker.py          # Tracker implementation
│   └── utils.py            # Tracker utilities
├── updater/                # OTA update system
│   └── ota_updater.py      # Update manager
└── utils/                  # Utility modules
    ├── connectivity.py     # Network connectivity
    └── dataloaders.py      # Data loading utilities
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- DeepSORT algorithm
- ByteTrack algorithm