import cv2
import time
import os
import numpy as np
import torch
import math
from pathlib import Path
from tracker_visualizer import draw_and_log_tracks, finalize_visualizer



def load_detection_model(model_path, device='cpu', logger=None):
    if logger:
        logger.info(f"Loading PyTorch model: {model_path}")
    try:
        import os
        import sys
        import torch
        import shutil
        import tempfile
        import subprocess
        import math  # Required for preprocessing calculations

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
                # (we'll remove it after the process ends to avoid issues with loaded modules)
                if temp_dir in sys.path:
                    sys.path.remove(temp_dir)

        # If all methods failed, raise an exception with all error messages
        if not method_success:
            raise RuntimeError(f"All model loading methods failed:\n" + "\n".join(error_messages))

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


def run(
        model_path,
        source=0,
        tracker=None,
        config=None,
        offline_queue=None,
        online_mode=True,
        logger=None,
        display=False,
        save_video=False,
        output_dir='output'):

    import time
    import cv2
    import numpy as np
    import torch
    import os

    # Initialize config if None
    if config is None:
        config = {}

    if save_video:
        os.makedirs(output_dir, exist_ok=True)

    device = config.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    model, preprocess = load_detection_model(model_path, device, logger)

    # Add debugging info about model
    if logger:
        logger.info(f"Model type: {type(model)}")
        if hasattr(model, 'names'):
            logger.info(f"Model classes: {model.names}")
        else:
            logger.warning("Model does not have 'names' attribute")

    try:
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            camera_index = int(source)
            if logger:
                logger.info(f"Opening camera {camera_index}")
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                if logger:
                    logger.warning(f"Camera {camera_index} failed to open, switching to fallback video.")
                fallback_video = config.get('input', {}).get('fallback_video', 'fallback.mp4')
                cap = cv2.VideoCapture(fallback_video)
                if logger:
                    logger.info(f"Using fallback video: {fallback_video}")
        else:
            if logger:
                logger.info(f"Opening video file {source}")
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            if logger:
                logger.error(f"Error opening video source {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = None
        if save_video:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_dir, f"tracking_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if logger:
                logger.info(f"Saving output to {output_path}")

        # Get confidence threshold from config with fallback to 0.15
        conf_threshold = config.get('detection', {}).get('confidence', 0.15)
        iou_threshold = config.get('detection', {}).get('iou_threshold', 0.45)
        detection_frequency = config.get('detection', {}).get('frequency', 1)

        # Update model confidence threshold if available
        if hasattr(model, 'conf'):
            model.conf = conf_threshold
            if logger:
                logger.info(f"Set model confidence threshold to {conf_threshold}")

        frame_count = 0
        detection_count = 0
        start_time = time.time()
        fps_display_interval = 30

        # Initialize performance stats
        total_inference_time = 0
        total_inference_frames = 0
        max_fps = 0
        min_fps = float('inf')

        while True:
            ret, frame = cap.read()
            if not ret:
                if logger:
                    logger.info("End of video stream")
                break

            frame_count += 1

            if frame_count % detection_frequency == 0:
                detection_count += 1
                display_frame = frame.copy()

                # Run inference and process detections
                try:
                    # Measure inference time
                    inference_start_time = time.time()

                    processed_frame = preprocess(frame)
                    detections = []

                    # Run inference with no_grad for better performance
                    with torch.no_grad():
                        results = model(processed_frame)

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

                    # Calculate inference time
                    inference_time = time.time() - inference_start_time
                    total_inference_time += inference_time
                    total_inference_frames += 1
                    current_fps = 1.0 / inference_time if inference_time > 0 else 0
                    max_fps = max(max_fps, current_fps)
                    min_fps = min(min_fps, current_fps) if current_fps > 0 else min_fps

                    # Log inference stats periodically
                    if frame_count % 100 == 0:
                        avg_fps = total_inference_frames / total_inference_time if total_inference_time > 0 else 0
                        if logger:
                            logger.info(
                                f"Inference stats: Avg FPS: {avg_fps:.2f}, Min: {min_fps:.2f}, Max: {max_fps:.2f}")

                    # Log detection results for all frames periodically
                    if logger and frame_count % 20 == 0:  # Log every 20th frame to avoid spam
                        logger.info(f"Frame {frame_count}: {len(detections)} detections found")
                        if len(detections) > 0:
                            # Log first few detections
                            for i, det in enumerate(detections[:3]):  # Show up to 3 detections
                                logger.info(
                                    f"  Detection {i + 1}: {det['class_name']}, conf={det['confidence']:.2f}, bbox={[int(x) for x in det['bbox']]}")

                except Exception as e:
                    if logger:
                        logger.error(f"Error processing detections: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    # Continue without detections
                    detections = []

                # Update tracker with detections if available
                tracks = []
                if tracker is not None:
                    try:
                        # Fix detection format if needed
                        for det in detections:
                            # Check if bbox is in format [x, y, w, h] instead of [x1, y1, x2, y2]
                            bbox = det['bbox']
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = bbox
                                # If x2, y2 are actually width and height, convert to x2, y2 coordinates
                                if x2 < x1 or y2 < y1:
                                    det['bbox'] = [x1, y1, x1 + x2, y1 + y2]

                        # Update tracker
                        tracks = tracker.update(detections, display_frame)

                        if logger and len(tracks) > 0 and frame_count % 20 == 0:
                            logger.info(f"Frame {frame_count}: {len(tracks)} active tracks")
                    except Exception as e:
                        if logger:
                            logger.error(f"Error updating tracker: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        tracks = []

                # Store data in offline queue if not in online mode
                if offline_queue is not None and not online_mode:
                    try:
                        offline_queue.add_frame_data({
                            'timestamp': time.time(),
                            'detections': detections,
                            'tracks': tracks
                        })
                    except Exception as e:
                        if logger:
                            logger.error(f"Error adding data to offline queue: {e}")

                # Update frame with detections and tracks - try/except in case api module is not available
                try:
                    from api import update_frame
                    update_frame(display_frame, detections, tracks)
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not use api.update_frame: {e}")
                    # Continue without API update

                # Draw detections and tracks on frame
                if display or save_video:
                    for det in detections:
                        try:
                            x1, y1, x2, y2 = map(int, det['bbox'])
                            label = f"{det['class_name']} {det['confidence']:.2f}"
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                        2)
                        except Exception as e:
                            if logger:
                                logger.error(f"Error drawing detection: {e}")

                    for track in tracks:
                        try:
                            track_id = track.get('id', 0)
                            # Draw tracks and log to CSV
                            if display or save_video:
                                draw_and_log_tracks(display_frame, tracks, model, frame_count)

                            x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 2)
                        except Exception as e:
                            if logger:
                                logger.error(f"Error drawing track: {e}")

                # Calculate and display FPS
                if frame_count % fps_display_interval == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps_measurement = fps_display_interval / elapsed if elapsed > 0 else 0
                    start_time = current_time
                    if logger:
                        logger.info(f"Processing FPS: {fps_measurement:.1f}")
                    if display or save_video:
                        cv2.putText(display_frame, f"FPS: {fps_measurement:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

                # Display and save frame
                if display:
                    cv2.imshow("Tracking", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        if logger:
                            logger.info("User terminated tracking")
                        break

                if save_video and writer is not None:
                    writer.write(display_frame)

            else:
                # Skip detection for this frame, but still display it
                if display:
                    cv2.imshow("Tracking", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        if logger:
                            logger.info("User terminated tracking")
                        break
                if save_video and writer is not None:
                    writer.write(frame)

        # Log final stats
        if logger and total_inference_frames > 0:
            avg_inference_time = total_inference_time / total_inference_frames
            avg_fps = total_inference_frames / total_inference_time if total_inference_time > 0 else 0
            logger.info(f"Final stats: Frames processed: {frame_count}, Inference frames: {total_inference_frames}")
            logger.info(f"Average inference time: {avg_inference_time * 1000:.2f}ms, Average FPS: {avg_fps:.2f}")
            logger.info(f"FPS range: Min: {min_fps:.2f}, Max: {max_fps:.2f}")


        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    except Exception as e:
        if logger:
            logger.error(f"Error during tracking: {e}")
            logger.exception(e)
        raise


if __name__ == "__main__":
    # Simple standalone test
    import argparse
    import yaml
    import logging
    import os

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DetectAndTrack")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Detection and Tracking Test")
    parser.add_argument("--model", default="yolov5s.pt", help="Path to model file")
    parser.add_argument("--source", default=0, help="Source (0 for webcam, path for video file)")
    parser.add_argument("--config", default="settings.yaml", help="Path to config file")
    parser.add_argument("--display", action="store_true", help="Display output")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--conf-thres", type=float, default=0.15, help="Confidence threshold")

    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            if logger:
                logger.info(f"Config loaded from {args.config}")
        except Exception as e:
            if logger:
                logger.error(f"Error loading config: {e}")
    else:
        if logger:
            logger.warning(f"Config file {args.config} not found, using default settings")

    # Set confidence threshold from args
    if not config.get('detection'):
        config['detection'] = {}
    config['detection']['confidence'] = args.conf_thres

    # Run detection and tracking
    run(
        model_path=args.model,
        source=args.source,
        config=config,
        logger=logger,
        display=args.display,
        save_video=args.save_video
    )
finalize_visualizer()
