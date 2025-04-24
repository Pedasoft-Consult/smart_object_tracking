# Detection and Tracking Module

## Overview
This module provides detection and tracking capabilities using YOLOv5 models. It handles model loading, inference, and integration with optional tracking systems.

## Recent Fixes and Improvements
The module has been updated with the following improvements:

1. **Fixed Detection Algorithm**
   - Corrected confidence threshold handling (now properly using the configured value)
   - Improved image preprocessing to maintain aspect ratio
   - Added proper setting of IoU threshold for newer YOLOv5 models
   - Enhanced preprocessing to work with different model versions

2. **Configuration Management**
   - Added command-line parameters for direct override of detection settings
   - Added proper error handling for missing configuration files
   - System now continues with default settings when config file is not found
   - Updated configuration file with optimized detection parameters
   - Changed detection frequency from 5 to 1 to process every frame

3. **Enhanced YOLOv5 Output Processing**
   - Completely rewrote the detection processing logic
   - Added support for multiple result formats:
     - Standard YOLOv5 Detections objects with xyxy attributes
     - Pandas-based outputs
     - Raw tensor outputs (with multiple tensor shapes)
     - List-based outputs

4. **Better Error Handling**
   - Improved error catching and logging throughout
   - Added specific error handling for the API module
   - Added debugging output for model results

5. **Performance Monitoring**
   - Added detailed performance statistics
   - Tracks min/max/average FPS
   - Reports inference times

## Usage
```python
# Using detect_and_track.py directly
python detect_and_track.py --model yolov5s.pt --source data/images/image.jpg --display --config config/settings.yaml

# Using main.py with confidence threshold override
python main.py --source car-detection.mp4 --display --save-video --confidence 0.2

# Using main.py with configuration file
python main.py --source car-detection.mp4 --display --save-video --config configs/settings.yaml
```

### detect_and_track.py Parameters
- `--model`: Path to YOLOv5 model file (.pt)
- `--source`: Input source (webcam index or video/image file path)
- `--config`: Configuration file path (YAML)
- `--display`: Flag to display output
- `--save-video`: Flag to save processing results as video

### main.py Parameters (Additional)
- `--confidence`: Override the detection confidence threshold
- `--iou-threshold`: Override the IoU threshold for NMS
- `--output-dir`: Specify output directory for saved videos
- `--tracker`: Choose tracker type ("deep_sort" or "byte_track")

## Recommended Configuration
The optimal settings for reliable detection:
- Confidence threshold: 0.2 (balances detection rate vs. false positives)
- IoU threshold: 0.45 (standard for YOLOv5)
- Detection frequency: 1 (process every frame)

## Default Configuration
If no configuration file is provided, the system uses these defaults:
- Device: CPU 
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Detection frequency: 1 (process every frame)

## Model Loading
The system attempts to load YOLOv5 models using multiple methods:
1. Direct torch.hub load with trust_repo=True
2. Specific YOLOv5 version (v6.2)
3. Cloning the repository and loading directly

## Dependencies
- PyTorch
- OpenCV
- NumPy
- PyYAML (for configuration files)
- Math (for preprocessing calculations)
- (Optional) Tracker module

## Key Fixes for Detecting Objects
If your system isn't detecting objects, check these key areas:

1. **Configuration Issues**:
   - Original code used hardcoded confidence of 0.1 instead of the configured value
   - Detection frequency of 5 was causing frames to be skipped
   - Extremely low confidence thresholds (0.01) may cause the model to filter detections

2. **Image Preprocessing**:
   - Original code didn't maintain aspect ratio during resizing
   - Fixed preprocessing properly handles different image shapes
   - BGR to RGB conversion is now handled correctly

3. **Model Threshold Settings**:
   - `model.conf` and `model.iou` are now set correctly for YOLOv5 models
   - Added proper handling for different YOLOv5 output formats

4. **Command-line Parameters**:
   - Added direct override abilities for confidence threshold
   - Added IoU threshold override
   - Simplified testing of different detection parameters

## Troubleshooting
If you encounter issues with model loading or detection processing:

1. **No detections found**: 
   - Try reducing the confidence threshold to 0.15-0.2 (use `--confidence 0.15`)
   - Ensure detection frequency is set to 1 to process every frame
   - Check if the model can detect the classes present in your video

2. **Detection performance issues**:
   - The preprocessing now maintains aspect ratio for better accuracy
   - Verify that the model's confidence threshold isn't being overridden in code
   - Check that IoU threshold is properly set for non-maximum suppression

3. **Other common issues**:
   - CUDA issues: If CUDA is not available, the system will fall back to CPU
   - Model loading failures: Multiple fallback methods are attempted
   - Output format issues: The system now handles different YOLOv5 output formats
   - Configuration errors: Default values are used when configuration is missing

For more details, enable logging to see detailed diagnostic information. Use the logging options in your configuration file to set the appropriate level:
```yaml
logging:
  level: "DEBUG"  # Change from INFO to DEBUG for more details
```