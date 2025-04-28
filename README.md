# Smart Object Tracking System

A real-time object detection and tracking system designed for edge devices with intermittent network connectivity and continuous learning capabilities.

## Features

- Real-time object detection using YOLOv5 and other models
- Multiple tracking algorithms (DeepSORT, ByteTrack)
- Online and offline operation modes with seamless switching
- Offline queue for storing detections when disconnected
- Over-the-air model updates
- REST API for remote monitoring and control
- Comprehensive model registry with support for multiple frameworks
- **Continuous Learning Framework**:
  - Annotation interface for correcting detections
  - Feedback collection and management
  - Automated model retraining using feedback data
  - Model evaluation and deployment pipeline

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

6. **Continuous Learning System**
   - Dataset management for training data
   - Annotation interface for correcting detections 
   - Feedback collection and processing
   - Automated model training with YOLOv5
   - Training status monitoring and evaluation
   - Automated model deployment

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

# Install TensorBoard for training visualization
pip install tensorboard
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

# Dataset settings
dataset:
  directory: "dataset"
  format: "yolo"
  class_file: "dataset/classes.txt"

# Feedback settings
feedback:
  directory: "feedback"
  min_items_process: 100
  min_items_retrain: 500

# Retraining settings
retraining:
  enabled: true
  check_interval: 3600  # Check every hour
  min_feedback_items: 500
  min_interval: 86400  # Minimum 1 day between retraining
  auto_deploy: true
  auto_start: true
```

## Dataset Preparation

Before training, you need to prepare your dataset in the YOLOv5 format:

```
dataset/
├── images/
│   ├── train/      # Training images (.jpg, .png)
│   ├── val/        # Validation images
│   └── test/       # Test images (optional)
├── labels/
│   ├── train/      # Training labels (.txt)
│   ├── val/        # Validation labels
│   └── test/       # Test labels (optional)
├── classes.txt     # List of class names (one per line)
└── dataset.yaml    # Dataset configuration
```

The `dataset.yaml` file should have the following format:

```yaml
# Dataset paths (relative to project root)
path: dataset
train: images/train
val: images/val
test: images/test  # optional

# Classes
nc: 3  # number of classes
names: ['person', 'car', 'bicycle']  # class names
```

You can use the dataset_manager.py to help manage your dataset:

```bash
# Create a dataset.yaml file
python -c "from dataset_manager import DatasetManager; DatasetManager('dataset').export_dataset_yaml()"
```

### Class ID Consistency

Each label file contains annotations in the YOLO format, where each line represents one object:
```
class_id x_center y_center width height
```

When preparing your dataset, make sure that:
1. The class IDs in your label files start from 0
2. The `nc` value in dataset.yaml matches the number of classes
3. All class IDs in your labels are less than the `nc` value

For example, if you have class IDs 0, 1, and 2, your dataset.yaml should have `nc: 3` and three class names.

You can use the provided `fix_dataset_simplified.py` script to automatically fix class ID issues:

```bash
python fix_dataset_simplified.py dataset
```

This script will:
- Scan all your label files to find the maximum class ID
- Update your dataset.yaml to have the correct number of classes (nc)
- Ensure class names list matches the required number of classes

## Usage

### Basic Tracking

Run the tracking system with default settings:

```bash
python main.py
```

### Command Line Options for Tracking

```bash
python main.py --source 0  # Use camera 0
python main.py --source video.mp4  # Use video file
python main.py --display  # Show visual output
python main.py --save-video  # Save output video
python main.py --tracker byte_track  # Use ByteTrack algorithm
```

### Running the API Server

The API server provides web interfaces and REST endpoints:

```bash
python main_integration.py --api
```

This starts the API server at http://localhost:5000 with the following interfaces:
- Web dashboard: http://localhost:5000/
- Annotation interface: http://localhost:5000/api/annotation/
- REST endpoints: http://localhost:5000/api/*

### Using the Annotation Interface

Access the annotation interface at http://localhost:5000/api/annotation/

The annotation interface allows you to:
- View and pause real-time detection results
- Add, edit, or delete bounding boxes
- Correct object classifications
- Submit corrected detections as feedback
- Monitor feedback statistics

## Model Training

### Training a New Model

Use the model_trainer.py script to train a new model:

```bash
# Train a YOLOv5s model for 100 epochs
python model_trainer.py --action train --model-type yolov5s --epochs 100 --batch-size 16 --img-size 640
```

For smaller datasets, use the enhanced model trainer with optimized settings:

```bash
# Train using enhanced trainer for small datasets
python model_trainer_enhanced.py --action train --model-type yolov5s --epochs 100 --batch-size 8 --img-size 640 --freeze 10 --lr 0.005 --patience 30 --use-local
```

Key enhanced training options:
- `--freeze N`: Freeze the first N layers to prevent overfitting on small datasets
- `--lr 0.005`: Lower learning rate for more stable training
- `--patience 30`: Early stopping patience (stops training if no improvement for 30 epochs)
- `--batch-size 8`: Smaller batch size for better generalization
- `--use-local`: Force local training instead of remote service

This will:
1. Clone the YOLOv5 repository if not already present
2. Train the model using your dataset
3. Register the trained model in the model registry
4. Export it to ONNX format for deployment

### Monitoring Training Progress

The training progress is logged to the console, but you can also use TensorBoard for visualization:

```bash
# Find your training ID (it will be output when you start training)
python model_trainer.py --action status

# Start TensorBoard to monitor the training
tensorboard --logdir models/[training_id]/weights
```

Then access TensorBoard at http://localhost:6006 in your browser.

### Training Recommendations

For best results:
1. **Complete all 100 epochs** — early stopping may prevent reaching optimal performance
2. **Ensure balanced classes** — have a similar number of examples for each class
3. **Use sufficient data** — aim for at least 500 images per class
4. **Monitor validation mAP** — to detect overfitting
5. **Use augmentation** — to improve model generalization
6. **Freeze early layers** — for small datasets, freeze the first 10-15 layers
7. **Adjust learning rate** — use lower learning rates (0.001-0.005) for small datasets
8. **Validate dataset structure** — ensure dataset.yaml and class IDs are correct

### Model Evaluation and Export

After training is complete, you can evaluate and export your model:

```bash
# List available models
python model_trainer.py --action list

# Evaluate a model (use the actual model ID from the list command)
python model_trainer.py --action evaluate --model-id model_1234567890_yolov5s

# Export a model to ONNX
python model_trainer.py --action export --model-id model_1234567890_yolov5s --format onnx
```

### Troubleshooting Model Training

#### Class ID Mismatch Issues

If you encounter errors like `Label class X exceeds nc=Y in dataset/dataset.yaml`, this means your dataset.yaml doesn't match your actual labels:

1. **Analyze label files and fix dataset.yaml**:
   ```bash
   # Use the included fix script
   python fix_dataset_simplified.py dataset
   ```

2. **Manual checks and fixes**:
   ```bash
   # Check for the highest class ID in your labels
   grep -r "" dataset/labels/ | cut -d " " -f 1 | sort -rn | head -1
   
   # Verify your dataset.yaml
   cat dataset/dataset.yaml
   
   # Verify your classes.txt
   cat dataset/classes.txt
   ```

3. **For more detailed analysis**:
   ```bash
   # Analyze and visualize your dataset
   python label_stats_and_fix.py --dataset dataset --visualize --update-yaml
   ```

4. **Use dataset manager to rebuild the YAML**:
   ```python
   from dataset_manager import DatasetManager
   dm = DatasetManager("dataset")
   dm.export_dataset_yaml()
   ```

#### Model File Not Found Issues

If you encounter "Model file not found" errors after training completes successfully, try these solutions:

1. **Locate the actual model files**:
   ```bash
   # Find all .pt files that were recently created
   find models -name "*.pt" -type f -mtime -1 | sort
   ```

2. **Disable Comet ML Integration**:
   The Comet ML integration can sometimes interfere with model saving. Either use:
   ```bash
   export COMET_MODE=disabled
   ```
   Or modify `model_trainer.py` to add:
   ```python
   # Add at the beginning of _train_local_thread method
   os.environ['COMET_MODE'] = 'disabled'
   ```

3. **Modify YOLOv5 command**:
   Add explicit save parameters:
   ```python
   # Add to cmd list in _train_local_thread method
   "--save-period", "1",  # Save every epoch
   "--exist-ok",          # Overwrite existing files
   "--nosave", "False",   # Ensure saving is enabled
   ```

4. **Enhanced model finder tool**:
   For persistent issues, use the provided diagnostic script:
   ```bash
   python model_finder_script.py models --training-id [training_id] --full-analysis
   ```
   This script will locate model files, analyze YOLOv5 output structure, and provide specific recommendations.

5. **Manual copy of model files**:
   If the script finds a model file in an unexpected location, you can manually copy it:
   ```bash
   # Copy found model to expected location
   cp "[found_model_path]" models/[training_id]/weights/best.pt
   ```

#### Other Common Training Issues

1. **CUDA out of memory**:
   - Reduce batch size (e.g., `--batch-size 8` or lower)
   - Use a smaller model variant (`--model-type yolov5n`)
   - Reduce image size (`--img-size 416`)

2. **Training stuck or slow progress**:
   - Make sure your dataset is properly structured
   - Check GPU utilization with `nvidia-smi -l 1`
   - Try clearing CUDA cache: `torch.cuda.empty_cache()`

3. **YOLOv5 repository issues**:
   - Try removing the YOLOv5 directory and let the trainer re-clone it
   - Specify a particular YOLOv5 version to avoid compatibility issues

4. **Validation set errors**:
   - Ensure you have a 'val' split with images and labels
   - If missing, the enhanced trainer will try to create one automatically
   - You can also manually create a validation set by copying files from train set

### Triggering Retraining Manually

Use the API endpoint to manually trigger retraining:

```bash
curl -X POST http://localhost:5000/api/retraining/trigger
```

Or use the web interface's "Trigger Retraining" button in the model management section.

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
| `/api/annotation/` | GET | Access annotation interface |
| `/api/annotation/submit` | POST | Submit corrected annotations |
| `/api/annotation/classes` | GET | Get available classes |
| `/api/feedback/stats` | GET | Get feedback statistics |
| `/api/training/status` | GET | Get training status |
| `/api/retraining/status` | GET | Get retraining scheduler status |
| `/api/retraining/trigger` | POST | Trigger retraining |

## Components

### Dataset Manager

The dataset manager handles training data:

```python
from dataset_manager import DatasetManager

# Initialize dataset manager
dataset_manager = DatasetManager("dataset")

# Add classes
dataset_manager.add_class("person")
dataset_manager.add_class("car")

# Add images and annotations
dataset_manager.add_from_file("image.jpg", "annotations.txt")

# Split dataset
dataset_manager.split_dataset(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

# Export dataset for training
dataset_manager.export_dataset_yaml()
```

### Model Trainer

The model trainer handles model training and export:

```python
from model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer("dataset", "models", config)

# Configure training
training_config = trainer.prepare_training_config(
    model_type="yolov5s",
    epochs=100,
    batch_size=16,
    img_size=640
)

# Start training
training_id = trainer.train(training_config)

# Check status
status = trainer.get_training_status(training_id)

# Evaluate trained model
metrics = trainer.evaluate(model_id="model_12345")

# Export model
exported_path = trainer.export_model(model_id="model_12345", format="onnx")
```

### Feedback Manager

The feedback manager collects and processes user corrections:

```python
from feedback_manager import FeedbackManager

# Initialize feedback manager
feedback_manager = FeedbackManager("feedback", dataset_manager)

# Add feedback
feedback_id = feedback_manager.add_feedback(
    frame_id="frame_123",
    original_detections=[{"bbox": [100, 100, 200, 200], "class_id": 0}],
    corrected_detections=[{"bbox": [110, 110, 210, 210], "class_id": 0}],
    image=frame
)

# Get feedback statistics
stats = feedback_manager.get_statistics()

# Process pending feedback
feedback_manager.process_pending_feedback()

# Trigger retraining
training_id = feedback_manager.trigger_retraining(model_trainer)
```

### Retraining Scheduler

The retraining scheduler automates the retraining process:

```python
from retraining_scheduler import RetrainingScheduler

# Initialize scheduler
scheduler = RetrainingScheduler(feedback_manager, model_trainer, config)

# Start scheduler
scheduler.start()

# Check status
status = scheduler.get_status()

# Force immediate retraining
training_id = scheduler.force_retrain()

# Stop scheduler
scheduler.stop()
```

## Helper Utilities

The system includes several utility scripts for troubleshooting and maintaining the training pipeline:

### Fix Dataset Classes

Fixes class ID issues in dataset.yaml:

```bash
python fix_dataset_simplified.py [dataset_dir]
```

### Analyze Label Statistics

Provides detailed statistics about dataset labels:

```bash
python label_stats_and_fix.py --dataset dataset --fix --visualize --update-yaml
```

### Find Model Files

Helps locate model files when standard paths fail:

```bash
python model_finder_script_enhanced.py models --training-id [training_id] --full-analysis
```

### Check Dataset YAML

Validates dataset.yaml against actual labels:

```bash
python check_dataset_yaml.py --dataset dataset
```

## Troubleshooting

### Common Issues

1. **API server not starting**: 
   - Make sure you're using `--api` flag with double dashes
   - Check if the port is already in use
   - Look for error messages in the server console log

2. **Annotation interface not accessible**:
   - Verify the correct URL: http://localhost:5000/api/annotation/
   - Check server logs for blueprint registration errors
   - Ensure annotation_ui.py exists and is properly integrated

3. **Model training fails**:
   - Check if YOLOv5 repository was properly cloned (look for models/yolov5)
   - Verify dataset structure and YAML configuration (use fix_dataset_simplified.py)
   - Check for sufficient disk space and GPU memory
   - Try running with smaller batch size if you encounter CUDA out of memory errors

4. **"No module named train" error**:
   - This happens when trying to run YOLOv5's train.py directly. Use model_trainer.py instead:
     ```bash
     python model_trainer.py --action train --model-type yolov5s --epochs 100
     ```

5. **Model not found for evaluation/export**:
   - Verify you have trained a model first
   - Use `--action list` to see available models
   - Use the exact model ID shown in the list
   - Use model_finder_script.py to locate models in non-standard locations

6. **Cannot access video feed**:
   - Check camera or video file permissions
   - Verify the correct source (camera index or file path)
   - Try an alternative camera or video file

7. **Missing model files after training**:
   - Check for Comet ML integration interference 
   - Use explicit save flags when calling YOLOv5
   - Use the model finder script to locate files saved in unexpected locations

8. **"Label class X exceeds nc=Y" error**:
   - This means your dataset.yaml doesn't match your actual label files
   - Run `python fix_dataset_simplified.py dataset` to fix this automatically

### Getting Help

If you encounter issues not covered here:
1. Check the server log for detailed error messages
2. Look for similar issues in the project repository
3. Submit a detailed bug report with logs and system information

## Project Structure

```
smart_object_tracking/
├── api.py                  # API server implementation
├── api_client.py           # API client library
├── annotation_ui.py        # Annotation interface
├── configs/                # Configuration files
│   └── settings.yaml       # Main configuration file
├── dataset_manager.py      # Dataset management
├── detect_and_track.py     # Core detection and tracking
├── enhanced_offline_queue.py # Enhanced offline queue
├── feedback_manager.py     # Feedback collection and management
├── fix_dataset_simplified.py # Fix dataset class issues
├── label_stats_and_fix.py  # Label analysis and fixing utility
├── main.py                 # Main entry point
├── main_integration.py     # Integration with continuous learning
├── model_exporter.py       # Model export functionality
├── model_evaluator.py      # Model evaluation capabilities
├── model_finder_script_enhanced.py # Find model files
├── model_trainer.py        # Model training system
├── model_trainer_enhanced.py # Enhanced trainer for small datasets
├── models/                 # Model storage
├── dataset/                # Dataset directory
│   ├── images/             # Training, validation, and test images
│   └── labels/             # Corresponding labels
├── offline_queue.py        # Basic offline queue
├── retraining_scheduler.py # Automated retraining scheduler
├── tests/                  # Test suite
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