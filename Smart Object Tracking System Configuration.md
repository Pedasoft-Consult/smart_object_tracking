# Smart Object Tracking System Configuration

# System information
system:
  name: "Smart Object Tracking System"
  version: "1.0.0"
  description: "Real-time object detection and tracking system for edge devices"

# Model settings
models:
  directory: "models"
  online_model: "yolov5s.pt"  # Model to use in online mode
  offline_model: "yolov5s-fp16.onnx"  # Model to use in offline mode
  registry_dir: "models/registry"  # Directory for model registry

# Input settings
input:
  default_source: 0  # Default camera (0 = primary webcam)
  max_resolution: [1280, 720]  # Maximum input resolution
  fps_target: 30  # Target FPS
  fallback_video: videos/person-bicycle-car-detection.mp4

# Detection settings
detection:
  confidence: 0.15  # Detection confidence threshold
  iou_threshold: 0.45  # IoU threshold for NMS
  frequency: 1  # Run detection every frame
  classes: null  # Filter specific classes (null = all classes)

# Tracking settings
tracking:
  tracker: "deep_sort"  # Tracker type: "deep_sort" or "byte_track"
  max_age: 30  # Maximum frames a track can be inactive before being removed
  min_hits: 3  # Minimum detections before track is displayed
  iou_threshold: 0.3  # IoU threshold for association
  max_feature_distance: 0.5  # Maximum feature distance for DeepSORT
  persistence: 30  # How long to keep tracks without detection
  feature_model: null  # Path to feature extraction model (null = use default features)

  # ByteTrack specific settings
  high_threshold: 0.6  # High confidence threshold for first association
  low_threshold: 0.1  # Low confidence threshold for second association

# Offline mode settings
offline:
  queue_directory: "queue"  # Directory for offline queue
  max_queue_size: 1000  # Maximum items in memory queue
  sync_interval: 3600  # Sync interval in seconds when going online (0 = manual sync)

# Update settings
updates:
  enabled: true  # Enable OTA updates
  check_url: "https://api.example.com/model-updates/check"  # URL to check for updates
  download_url: "https://storage.example.com/models/"  # Base URL for model downloads
  interval: 86400  # Check interval in seconds (default: 1 day)

# Logging settings
logging:
  level: "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR)
  directory: "logs"  # Log directory
  console_output: true  # Display logs on console
  file_output: true  # Save logs to file
  max_log_size: 10485760  # Maximum log file size (10 MB)
  max_log_files: 10  # Maximum number of log files to keep

# Display settings
display:
  show_fps: true  # Show FPS counter
  show_confidence: true  # Show confidence scores
  show_labels: true  # Show class labels
  show_tracks: true  # Show tracking IDs

# Device settings
device: "cpu"  # Inference device: "cpu" or "cuda"

# Dataset settings
dataset:
  directory: "dataset"  # Directory for training data
  format: "yolo"  # Dataset format (yolo, coco, etc.)
  class_file: "dataset/classes.txt"  # Path to class definitions
  split_ratio: [0.7, 0.2, 0.1]  # Train/val/test split ratios

# Training settings
training:
  default_model_type: "yolov5s"  # Default base model for training
  default_epochs: 50  # Default number of epochs
  default_batch_size: 16  # Default batch size
  default_img_size: 640  # Default image size
  auto_export: true  # Automatically export to ONNX after training
  save_period: 1  # Save checkpoint every N epochs
  patience: 20  # Epochs to wait before early stopping
  workers: 4  # Number of dataloader workers
  disable_comet: true  # Disable Comet ML integration
  remote_api_endpoint: "https://api.example.com/training"  # API endpoint for remote training
  remote_api_key: "your_api_key_here"  # API key for remote training

  # Dataset-specific configurations
  dataset_configs:
    # Configuration for small datasets
    small_dataset:
      lr: 0.01  # Learning rate for small datasets
      batch_size: 8  # Smaller batch size for small datasets
      epochs: 100  # More epochs for small datasets
      freeze_layers: 10  # Freeze more layers for small datasets
      patience: 30  # More patience for early stopping
      augmentation: "heavy"  # Heavy augmentation for small datasets
      evolve_hyp: true  # Use hyperparameter evolution
      mixup: 0.25  # Add mixup for small datasets
      mosaic: 1.0  # Full mosaic augmentation
      copy_paste: 0.1  # Add copy-paste for small datasets
    
    # Configuration for large datasets
    large_dataset:
      lr: 0.005  # Lower learning rate for larger datasets
      batch_size: 32  # Larger batch size for larger datasets
      epochs: 50  # Fewer epochs needed for large datasets
      freeze_layers: 5  # Freeze fewer layers
      patience: 15  # Less patience needed
      augmentation: "medium"  # Medium augmentation is often sufficient
      evolve_hyp: false  # Less need for hyperparameter evolution
      mixup: 0.1  # Less mixup needed
      mosaic: 0.8  # Less mosaic needed

  # YOLOv5 hyperparameters (can be fine-tuned)
  hyperparameters:
    lr0: 0.01  # Initial learning rate
    lrf: 0.1  # Final learning rate (lr0 * lrf)
    momentum: 0.937  # SGD momentum
    weight_decay: 0.0005  # Optimizer weight decay
    warmup_epochs: 3.0  # Warmup epochs
    warmup_momentum: 0.8  # Warmup momentum
    warmup_bias_lr: 0.1  # Warmup bias learning rate
    box: 0.05  # Box loss weight
    cls: 0.5  # Class loss weight
    cls_pw: 1.0  # Class positive weight
    obj: 1.0  # Object loss weight
    obj_pw: 1.0  # Object positive weight
    fl_gamma: 0.0  # Focal loss gamma
    hsv_h: 0.015  # Hue augmentation
    hsv_s: 0.7  # Saturation augmentation
    hsv_v: 0.4  # Value augmentation
    degrees: 0.0  # Rotation augmentation
    translate: 0.1  # Translation augmentation
    scale: 0.5  # Scale augmentation
    shear: 0.0  # Shear augmentation
    perspective: 0.0  # Perspective augmentation
    flipud: 0.0  # Vertical flip probability
    fliplr: 0.5  # Horizontal flip probability
    mosaic: 1.0  # Mosaic augmentation
    mixup: 0.0  # Mixup augmentation
    copy_paste: 0.0  # Copy-paste augmentation

# Feedback settings
feedback:
  directory: "feedback"  # Directory for user feedback
  min_items_process: 100  # Minimum feedback items to process
  min_items_retrain: 500  # Minimum feedback items to trigger retraining
  auto_process: true  # Automatically process feedback

# Retraining settings
retraining:
  enabled: true  # Enable automatic retraining
  check_interval: 3600  # Check interval in seconds
  min_feedback_items: 500  # Minimum feedback items to trigger retraining
  min_interval: 86400  # Minimum interval between retrainings (seconds)
  auto_deploy: true  # Automatically deploy retrained models
  auto_start: true  # Automatically start retraining when criteria met