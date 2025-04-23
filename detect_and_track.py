#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Detection and tracking module.
Handles detection model loading, inference, and tracking.
"""
import cv2
import time
import os
import numpy as np
from pathlib import Path

def load_detection_model(model_path, device='cpu', logger=None):
    if logger:
        logger.info(f"Loading PyTorch model: {model_path}")
    try:
        import os
        import sys
        original_path = sys.path.copy()
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root in sys.path:
                sys.path.remove(project_root)
            utils_module = sys.modules.pop('utils', None)
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            model.to(device)
            if utils_module is not None:
                sys.modules['utils'] = utils_module
        finally:
            sys.path = original_path

        def preprocess(image):
            import torch
            if isinstance(image, np.ndarray):
                if image.shape[2] == 3:
                    image = image[:, :, ::-1].copy()
                image = cv2.resize(image, (640, 640))
                image = torch.from_numpy(image).to(device)
                image = image.permute(2, 0, 1).float() / 255.0
                if image.ndimension() == 3:
                    image = image.unsqueeze(0)
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

    if save_video:
        os.makedirs(output_dir, exist_ok=True)

    device = config.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    model, preprocess = load_detection_model(model_path, device, logger)

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

        conf_threshold = config.get('detection', {}).get('confidence', 0.25)
        iou_threshold = config.get('detection', {}).get('iou_threshold', 0.45)
        detection_frequency = config.get('detection', {}).get('frequency', 1)

        frame_count = 0
        detection_count = 0
        start_time = time.time()
        fps_display_interval = 30

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
                processed_frame = preprocess(frame)
                results = model(processed_frame)
                detections = []

                if hasattr(results, 'xyxy'):
                    for det in results.xyxy[0]:
                        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                        if conf >= conf_threshold:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': model.names[int(cls)]
                            })

                if tracker is not None:
                    tracks = tracker.update(detections, display_frame)
                else:
                    tracks = []

                if offline_queue is not None and not online_mode:
                    offline_queue.add_frame_data({
                        'timestamp': time.time(),
                        'detections': detections,
                        'tracks': tracks
                    })

                if display or save_video:
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    for track in tracks:
                        track_id = track.get('id', 0)
                        x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if frame_count % fps_display_interval == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps_measurement = fps_display_interval / elapsed if elapsed > 0 else 0
                    start_time = current_time
                    if display or save_video:
                        cv2.putText(display_frame, f"FPS: {fps_measurement:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if display:
                    cv2.imshow("Tracking", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        if logger:
                            logger.info("User terminated tracking")
                        break

                if save_video and writer is not None:
                    writer.write(display_frame)

            else:
                if display:
                    cv2.imshow("Tracking", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        if logger:
                            logger.info("User terminated tracking")
                        break
                if save_video and writer is not None:
                    writer.write(frame)

        cap.release()
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

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DetectAndTrack")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Detection and Tracking Test")
    parser.add_argument("--model", default="yolov5s.pt", help="Path to model file")
    parser.add_argument("--source", default=0, help="Source (0 for webcam, path for video file)")
    parser.add_argument("--config", default="../configs/settings.yaml", help="Path to config file")
    parser.add_argument("--display", action="store_true", help="Display output")
    parser.add_argument("--save-video", action="store_true", help="Save output video")

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {}

    # Run detection and tracking
    run(
        model_path=args.model,
        source=args.source,
        config=config,
        logger=logger,
        display=args.display,
        save_video=args.save_video
    )