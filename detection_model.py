import cv2
import os
import sys
import torch
import shutil
import tempfile
import subprocess
import math
import numpy as np
import logging


def load_detection_model(model_path, device='cpu', logger=None):
    """
    Load a YOLOv5 detection model using multiple fallback methods.

    Args:
        model_path (str): Path to the model file
        device (str): Device to run inference on ('cpu' or 'cuda')
        logger: Logger object for logging information

    Returns:
        tuple: (model, preprocess_function)
    """
    if logger:
        logger.info(f"Loading PyTorch model: {model_path}")
    try:
        # Save original path and modules
        original_path = sys.path.copy()
        original_modules = dict(sys.modules)

        # Try multiple methods to load YOLOv5
        method_success = False
        error_messages = []

        # Method 1: Try direct torch.hub load with trust_repo=True
        if not method_success:
            try:
                # Temporarily disable deletion to avoid directory errors
                original_delete_dir = torch.hub._get_torch_home
                torch.hub._get_torch_home = lambda: os.path.expanduser("~/torch_hub_temp")

                # Remove problematic modules if they exist
                for key in list(sys.modules.keys()):
                    if key.startswith('models.') or key == 'utils':
                        sys.modules.pop(key, None)

                model = torch.hub.load('ultralytics/yolov5', 'custom',
                                       path=model_path,
                                       force_reload=True,
                                       trust_repo=True)

                # Set confidence threshold directly
                if hasattr(model, 'conf'):
                    if logger:
                        logger.info(f"Setting model confidence threshold to 0.15")
                    model.conf = 0.15  # Set to lower threshold

                model.to(device)
                method_success = True
                if logger:
                    logger.info("Successfully loaded model using Method 1: direct torch.hub")
                    if hasattr(model, 'non_max_suppression'):
                        logger.info("Model has non_max_suppression method, will use it directly")
            except Exception as e1:
                error_messages.append(f"Method 1 failed: {str(e1)}")
                if logger:
                    logger.warning(f"Method 1 load attempt failed: {e1}")
            finally:
                # Restore original torch.hub function
                torch.hub._get_torch_home = original_delete_dir

        # Method 2: Try with specific YOLOv5 version
        if not method_success:
            try:
                # Reset modules that might have been partially loaded
                for key in list(sys.modules.keys()):
                    if key not in original_modules and ('models' in key or key == 'utils'):
                        sys.modules.pop(key, None)

                model = torch.hub.load('ultralytics/yolov5:v6.2',
                                       'custom',
                                       path=model_path,
                                       trust_repo=True)

                # Set confidence threshold directly
                if hasattr(model, 'conf'):
                    if logger:
                        logger.info(f"Setting model confidence threshold to 0.15")
                    model.conf = 0.15  # Set to lower threshold

                model.to(device)
                method_success = True
                if logger:
                    logger.info("Successfully loaded model using Method 2: versioned torch.hub")
                    if hasattr(model, 'non_max_suppression'):
                        logger.info("Model has non_max_suppression method, will use it directly")
            except Exception as e2:
                error_messages.append(f"Method 2 failed: {str(e2)}")
                if logger:
                    logger.warning(f"Method 2 load attempt failed: {e2}")

        # Method 3: Clone the repo and load directly
        if not method_success:
            temp_dir = None
            try:
                # Create temp directory for YOLOv5 clone
                temp_dir = tempfile.mkdtemp()
                if logger:
                    logger.info(f"Cloning YOLOv5 to temp directory: {temp_dir}")

                # Clone the repository
                subprocess.check_call(['git', 'clone', '--depth', '1',
                                       'https://github.com/ultralytics/yolov5.git', temp_dir])

                # Install requirements
                requirements_file = os.path.join(temp_dir, 'requirements.txt')
                if os.path.exists(requirements_file):
                    try:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_file])
                    except Exception as req_e:
                        if logger:
                            logger.warning(f"Failed to install requirements: {req_e}")

                # Add to path and clear conflicting modules
                sys.path.insert(0, temp_dir)
                for key in list(sys.modules.keys()):
                    if key not in original_modules and ('models' in key or key == 'utils'):
                        sys.modules.pop(key, None)

                # Import directly from the cloned repo
                from models.experimental import attempt_load
                model = attempt_load(model_path, device=device)

                # Set confidence threshold
                if hasattr(model, 'conf'):
                    if logger:
                        logger.info(f"Setting model confidence threshold to 0.15")
                    model.conf = 0.15  # Set to lower threshold

                method_success = True
                if logger:
                    logger.info("Successfully loaded model using Method 3: local clone")
                    if hasattr(model, 'non_max_suppression'):
                        logger.info("Model has non_max_suppression method, will use it directly")
            except Exception as e3:
                error_messages.append(f"Method 3 failed: {str(e3)}")
                if logger:
                    logger.error(f"Method 3 load attempt failed: {e3}")
            finally:
                # Restore original path but keep temp_dir for now
                if temp_dir in sys.path:
                    sys.path.remove(temp_dir)

        # If all methods failed, raise an exception with all error messages
        if not method_success:
            raise RuntimeError(f"All model loading methods failed:\n" + "\n".join(error_messages))

        # Create preprocessing function based on the loaded model
        def preprocess(image):
            """
            Preprocess an image for YOLOv5 inference
            """
            if isinstance(image, np.ndarray):
                # Keep aspect ratio when resizing
                h, w = image.shape[:2]
                img_copy = image.copy()  # Make a copy to avoid modifying the original

                # Apply preprocessing based on model requirements
                if hasattr(model, 'model') and hasattr(model.model, 'stride'):
                    # For newer YOLOv5 models
                    stride = int(model.model.stride.max())
                    img_size = max(640, math.ceil(max(h, w) / stride) * stride)

                    # Maintain aspect ratio and pad
                    if h > w:
                        new_h, new_w = img_size, int(w * img_size / h)
                    else:
                        new_h, new_w = int(h * img_size / w), img_size

                    img_copy = cv2.resize(img_copy, (new_w, new_h))

                    # Convert BGR to RGB
                    if img_copy.shape[2] == 3:
                        img_copy = img_copy[:, :, ::-1].copy()  # BGR to RGB

                    # Convert to tensor and normalize
                    img_copy = torch.from_numpy(img_copy).to(device)
                    img_copy = img_copy.permute(2, 0, 1).float() / 255.0

                    # Add batch dimension if needed
                    if img_copy.ndimension() == 3:
                        img_copy = img_copy.unsqueeze(0)
                else:
                    # Simple resize for older models or fallback
                    if img_copy.shape[2] == 3:
                        img_copy = img_copy[:, :, ::-1].copy()  # BGR to RGB
                    img_copy = cv2.resize(img_copy, (640, 640))
                    img_copy = torch.from_numpy(img_copy).to(device)
                    img_copy = img_copy.permute(2, 0, 1).float() / 255.0
                    if img_copy.ndimension() == 3:
                        img_copy = img_copy.unsqueeze(0)

                return img_copy
            return image

        if logger:
            logger.info(f"Model loaded successfully on {device}")
        return model, preprocess

    except Exception as e:
        if logger:
            logger.error(f"Error loading PyTorch model: {e}")
        raise


def process_detections(model, results, conf_threshold, iou_threshold, logger=None, frame_count=0):
    """
    Process detection results from the model into a standardized format

    Args:
        model: The detection model
        results: The raw results from model inference
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        logger: Logger for debug information
        frame_count: Current frame number for logging

    Returns:
        list: List of detection dictionaries with bbox, confidence, class_id, class_name
    """
    detections = []

    try:
        # Debug raw model output for first few frames
        if frame_count <= 2 and logger:
            logger.info(f"Raw YOLO output (type: {type(results)}): {results}")

            # Log detailed tensor info
            if isinstance(results, tuple) and len(results) > 0:
                logger.info(f"Results tuple has {len(results)} elements")
                for i, elem in enumerate(results):
                    if isinstance(elem, torch.Tensor):
                        logger.info(f"Element {i} is tensor with shape {elem.shape}")
                        # If tensor has values > 0.15 confidence
                        if elem.dim() >= 2 and elem.shape[1] > 4:
                            # Get max confidence in this tensor
                            conf_col = 4  # Assuming confidence is at index 4
                            if conf_col < elem.shape[1] and elem[:, conf_col].max() > 0.15:
                                logger.info(f"Element {i} has high confidence detections")

        # Method 1: Try using model's built-in NMS if available
        if hasattr(model, 'non_max_suppression'):
            try:
                # Use the model's native non_max_suppression method
                # This is the most reliable method for YOLOv5
                preds = model.non_max_suppression(
                    results,
                    conf_thres=conf_threshold,
                    iou_thres=iou_threshold
                )

                if isinstance(preds, list) and len(preds) > 0:
                    pred = preds[0]  # First image in batch
                    for det in pred:
                        if len(det) >= 6:  # Should be [x1, y1, x2, y2, conf, cls]
                            x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
                            if conf >= conf_threshold:
                                class_id = int(cls_id)
                                class_name = model.names[class_id] if hasattr(model,
                                                                              'names') and class_id in model.names else f"class_{class_id}"
                                detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(conf),
                                    'class_id': class_id,
                                    'class_name': class_name
                                })
                if logger and frame_count <= 3:
                    logger.info(
                        f"Using model's non_max_suppression: found {len(detections)} detections with conf>{conf_threshold}")
            except Exception as e:
                if logger:
                    logger.error(f"Error using model's non_max_suppression: {e}")
                # Fall back to manual processing

        # Process results based on format if non_max_suppression failed or isn't available
        if len(detections) == 0:
            # Method 2: Process YOLOv5 output format (xyxy, conf, cls)
            if hasattr(results, 'xyxy'):
                pred = results.xyxy[0]  # Get predictions for first image
                for i in range(pred.shape[0]):
                    det = pred[i].cpu().numpy()
                    x1, y1, x2, y2, conf, cls_id = det
                    if conf >= conf_threshold:
                        class_id = int(cls_id)
                        class_name = model.names[class_id] if hasattr(model,
                                                                      'names') and class_id in model.names else f"class_{class_id}"
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class_id': class_id,
                            'class_name': class_name
                        })
                if logger and frame_count <= 3:
                    logger.info(f"Processed xyxy format: found {len(detections)} detections")

            # Method 3: Process pandas format
            elif hasattr(results, 'pandas'):
                preds = results.pandas().xyxy[0]
                for idx, row in preds.iterrows():
                    if row['confidence'] >= conf_threshold:
                        detections.append({
                            'bbox': [float(row['xmin']), float(row['ymin']), float(row['xmax']),
                                     float(row['ymax'])],
                            'confidence': float(row['confidence']),
                            'class_id': int(row['class']),
                            'class_name': row['name']
                        })
                if logger and frame_count <= 3:
                    logger.info(f"Processed pandas format: found {len(detections)} detections")

            # Method 4: Process tuple output (common in YOLOv5)
            elif isinstance(results, tuple) and len(results) >= 1:
                # Try to use the first element which is typically the detection tensor
                pred = results[0]
                if isinstance(pred, torch.Tensor):
                    # Process based on shape
                    if pred.dim() == 3 and pred.shape[0] == 1:  # [1, num_detections, features]
                        pred = pred[0]  # Remove batch dimension

                    if pred.dim() == 2:  # [num_detections, features]
                        # YOLOv5 output format: center_x, center_y, width, height, confidence, class_scores...
                        num_classes = pred.shape[1] - 5

                        # Filter by confidence
                        conf_mask = pred[:, 4] >= conf_threshold
                        if conf_mask.sum() > 0:
                            filtered_pred = pred[conf_mask]

                            # For each detection
                            for i in range(filtered_pred.shape[0]):
                                det = filtered_pred[i]

                                # Get box coordinates
                                x_center, y_center, width, height = det[0:4].cpu().numpy()

                                # Convert to xyxy format
                                x1 = float(x_center - width / 2)
                                y1 = float(y_center - height / 2)
                                x2 = float(x_center + width / 2)
                                y2 = float(y_center + height / 2)

                                # Get confidence and class
                                conf = float(det[4].cpu().numpy())

                                # Get class ID (index of max probability in class scores)
                                if num_classes > 1:
                                    class_scores = det[5:].cpu().numpy()
                                    class_id = int(np.argmax(class_scores))
                                else:
                                    class_id = int(det[5].cpu().numpy())

                                # Create detection dictionary
                                class_name = model.names[class_id] if hasattr(model,
                                                                              'names') and class_id in model.names else f"class_{class_id}"
                                detections.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': conf,
                                    'class_id': class_id,
                                    'class_name': class_name
                                })

                    if logger and frame_count <= 3:
                        logger.info(f"Processed tuple tensor: found {len(detections)} detections")

            # Method 5: Process standard tensor format
            elif isinstance(results, torch.Tensor):
                # Try to process as tensor output
                try:
                    # Standardize shape
                    if results.dim() == 3 and results.shape[0] == 1:  # [1, num_detections, features]
                        results = results[0]

                    if results.dim() == 2 and results.shape[1] >= 6:  # [num_detections, features]
                        # Assuming format is: x1, y1, x2, y2, conf, class_id
                        # Filter by confidence
                        conf_mask = results[:, 4] >= conf_threshold
                        filtered_results = results[conf_mask]

                        for i in range(filtered_results.shape[0]):
                            det = filtered_results[i].cpu().numpy()
                            x1, y1, x2, y2, conf, cls_id = det[:6]

                            class_id = int(cls_id)
                            class_name = model.names[class_id] if hasattr(model,
                                                                          'names') and class_id in model.names else f"class_{class_id}"

                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class_id': class_id,
                                'class_name': class_name
                            })
                except Exception as e:
                    if logger:
                        logger.error(f"Error processing tensor output: {e}")

            # Method 6: Handle list output
            elif isinstance(results, list) and len(results) > 0:
                pred = results[0]
                if isinstance(pred, torch.Tensor):
                    if pred.dim() == 2 and pred.shape[1] >= 6:  # [num_detections, xyxy+conf+class]
                        for i in range(pred.shape[0]):
                            det = pred[i].cpu().numpy()
                            x1, y1, x2, y2, conf, cls_id = det[:6]
                            if conf >= conf_threshold:
                                class_id = int(cls_id)
                                class_name = model.names[class_id] if hasattr(model,
                                                                              'names') and class_id in model.names else f"class_{class_id}"
                                detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(conf),
                                    'class_id': class_id,
                                    'class_name': class_name
                                })

        # Log detection count
        if logger and frame_count <= 2:
            logger.info(f"Model returned results of type: {type(results)}")
            logger.info(f"Detections found: {len(detections)}")

    except Exception as e:
        if logger:
            logger.error(f"Error processing detections: {e}")
            import traceback
            logger.error(traceback.format_exc())
        # Return empty detections on error
        detections = []

    return detections


def format_detections_for_tracker(detections, model, logger=None):
    """
    Format detections for the tracker

    Args:
        detections: List of detection dictionaries
        model: The detection model (for class name lookup)
        logger: Logger for debug info

    Returns:
        list: Properly formatted detections for the tracker
    """
    formatted_detections = []

    try:
        for det in detections:
            if isinstance(det, dict) and 'bbox' in det:
                # Check if bbox is valid and fix if needed
                bbox = det['bbox']
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # Check if x2, y2 are width/height instead of coordinates
                    if x2 < x1 or y2 < y1:
                        det['bbox'] = [x1, y1, x1 + x2, y1 + y2]
                    formatted_detections.append(det)
            elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 4:
                # Convert to dictionary format
                x1, y1, x2, y2 = det[:4]
                confidence = det[4] if len(det) > 4 else 0.0
                class_id = int(det[5]) if len(det) > 5 else 0
                formatted_detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': model.names[class_id] if hasattr(model,
                                                                   'names') and class_id in model.names else f"class_{class_id}"
                })
    except Exception as e:
        if logger:
            logger.error(f"Error formatting detections for tracker: {e}")

    return formatted_detections