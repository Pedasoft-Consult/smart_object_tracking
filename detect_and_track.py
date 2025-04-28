import cv2
import time
import os
import numpy as np
import torch
import logging
from pathlib import Path

# Import from the other modules
from detection_model import load_detection_model, process_detections, format_detections_for_tracker
from tracking_utils import (
    debug_dump_tracking_state,
    draw_detections,
    draw_tracks,
    draw_fps,
    draw_and_log_tracks,
    finalize_visualizer
)


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
    """
    Main function for running detection and tracking

    Args:
        model_path (str): Path to the model file
        source: Video source (0 for webcam, or path to video file)
        tracker: Object tracker instance
        config (dict): Configuration dictionary
        offline_queue: Queue for storing offline data
        online_mode (bool): Whether to run in online mode
        logger: Logger for logging information
        display (bool): Whether to display output
        save_video (bool): Whether to save output video
        output_dir (str): Directory for saving output

    Returns:
        bool: Success status
    """
    # Initialize config if None
    if config is None:
        config = {}

    if save_video:
        os.makedirs(output_dir, exist_ok=True)

    # Set device for inference
    device = config.get('device', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        if logger:
            logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Load the detection model
    try:
        model, preprocess = load_detection_model(model_path, device, logger)

        # Add debugging info about model
        if logger:
            logger.info(f"Model type: {type(model)}")
            if hasattr(model, 'names'):
                logger.info(f"Model classes: {model.names}")
            else:
                logger.warning("Model does not have 'names' attribute")
    except Exception as e:
        if logger:
            logger.error(f"Failed to load model: {e}")
        return False

    try:
        # Initialize video capture
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
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize video writer if needed
        writer = None
        if save_video:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_dir, f"tracking_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if logger:
                logger.info(f"Saving output to {output_path}")

        # Get detection parameters from config
        conf_threshold = config.get('detection', {}).get('confidence', 0.15)
        iou_threshold = config.get('detection', {}).get('iou_threshold', 0.45)
        detection_frequency = config.get('detection', {}).get('frequency', 1)

        # Update model confidence threshold if available
        if hasattr(model, 'conf'):
            model.conf = conf_threshold
            if logger:
                logger.info(f"Set model confidence threshold to {conf_threshold}")

        # Initialize counters and timers
        frame_count = 0
        detection_count = 0
        start_time = time.time()
        fps_display_interval = 30

        # Initialize performance stats
        total_inference_time = 0
        total_inference_frames = 0
        max_fps = 0
        min_fps = float('inf')

        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                if logger:
                    logger.info("End of video stream")
                break

            frame_count += 1

            # Run detection on specified frames based on detection_frequency
            if frame_count % detection_frequency == 0:
                detection_count += 1
                display_frame = frame.copy()

                # Run inference and process detections
                try:
                    # Measure inference time
                    inference_start_time = time.time()

                    # Preprocess frame
                    processed_frame = preprocess(frame)

                    # Run inference with no_grad for better performance
                    with torch.no_grad():
                        results = model(processed_frame)

                        # Process detections using helper function
                        detections = process_detections(
                            model,
                            results,
                            conf_threshold,
                            iou_threshold,
                            logger,
                            frame_count
                        )

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

                    # Log detection results periodically
                    if logger and frame_count % 20 == 0:  # Log every 20th frame to avoid spam
                        logger.info(f"Frame {frame_count}: {len(detections)} detections found")
                        for i, det in enumerate(detections[:3]):
                            logger.info(f"DETECTION DEBUG {i + 1}: {det}")

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
                        # Format detections for tracker
                        formatted_detections = format_detections_for_tracker(detections, model, logger)

                        # Update tracker with properly formatted detections
                        if formatted_detections:
                            try:
                                tracks = tracker.update(formatted_detections, display_frame)
                            except Exception as e:
                                if logger:
                                    logger.error(f"Error updating tracker: {e}")
                                    import traceback
                                    logger.error(traceback.format_exc())
                                debug_dump_tracking_state(e, [] if 'tracks' not in locals() else tracks,
                                                          formatted_detections, display_frame)
                                tracks = []
                        else:
                            if detections:
                                if logger:
                                    logger.warning(
                                        f"No valid detections to pass to tracker from {len(detections)} detections")
                            tracks = []

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

                # Update frame with detections and tracks via API if available
                try:
                    from api import update_frame
                    update_frame(frame, detections=detections, tracks=tracker.get_active_tracks())
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not use api.update_frame: {e}")
                    # Continue without API update

                # Draw detections and tracks on frame
                if display or save_video:
                    # Draw detections
                    display_frame = draw_detections(display_frame, detections)

                    # Draw tracks
                    if tracks and len(tracks) > 0:
                        try:
                            # Draw all tracks at once using the tracker_visualizer
                            display_frame = draw_and_log_tracks(display_frame, tracks, model, frame_count)
                        except Exception as e:
                            if logger:
                                logger.error(f"Error drawing tracks: {e}")
                            # Fallback - draw tracks manually
                            display_frame = draw_tracks(display_frame, tracks)

                    # Calculate and display FPS
                    current_fps = 0
                    if frame_count % fps_display_interval == 0:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        current_fps = fps_display_interval / elapsed if elapsed > 0 else 0
                        start_time = current_time
                        if logger:
                            logger.info(f"Processing FPS: {current_fps:.1f}")

                    # Add FPS to display frame
                    if display or save_video:
                        display_frame = draw_fps(display_frame, current_fps)

                    # Display and save frame
                    if display:
                        try:
                            cv2.imshow("Tracking", display_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == 27:  # ESC key
                                if logger:
                                    logger.info("User terminated tracking")
                                break
                        except Exception as e:
                            if logger:
                                logger.error(f"Error displaying frame: {e}")

                    if save_video and writer is not None:
                        try:
                            writer.write(display_frame)
                        except Exception as e:
                            if logger:
                                logger.error(f"Error writing video frame: {e}")

            else:
                # Skip detection for this frame, but still display it
                if display:
                    try:
                        cv2.imshow("Tracking", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC key
                            if logger:
                                logger.info("User terminated tracking")
                            break
                    except Exception as e:
                        if logger:
                            logger.error(f"Error displaying frame: {e}")

                if save_video and writer is not None:
                    try:
                        writer.write(frame)
                    except Exception as e:
                        if logger:
                            logger.error(f"Error writing video frame: {e}")

        # Log final stats
        if logger and total_inference_frames > 0:
            avg_inference_time = total_inference_time / total_inference_frames
            avg_fps = total_inference_frames / total_inference_time if total_inference_time > 0 else 0
            logger.info(f"Final stats: Frames processed: {frame_count}, Inference frames: {total_inference_frames}")
            logger.info(f"Average inference time: {avg_inference_time * 1000:.2f}ms, Average FPS: {avg_fps:.2f}")
            logger.info(f"FPS range: Min: {min_fps:.2f}, Max: {max_fps:.2f}")

        # Clean up resources
        try:
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            # Release the video capture object
            cap.release()
        except Exception as e:
            if logger:
                logger.error(f"Error during cleanup: {e}")

        return True  # Success

    except Exception as e:
        if logger:
            logger.error(f"Error during tracking: {e}")
            import traceback
            logger.error(traceback.format_exc())

        # Attempt to clean up resources even after error
        try:
            if 'cap' in locals() and cap is not None:
                cap.release()
            if 'writer' in locals() and writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        except:
            pass

        # Return failure
        return False


if __name__ == "__main__":
    # Simple standalone test
    import argparse
    import yaml
    import logging
    import os
    from tracker.tracker import ObjectTracker

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("DetectAndTrack")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Detection and Tracking Test")
    parser.add_argument("--model", default="yolov5s.pt", help="Path to model file")
    parser.add_argument("--source", default=0, help="Source (0 for webcam, path for video file)")
    parser.add_argument("--config", default="configs/settings.yaml", help="Path to config file")
    parser.add_argument("--display", action="store_true", help="Display output")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--conf-thres", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--tracker", choices=["deep_sort", "byte_track"], default="byte_track", help="Tracker type")
    parser.add_argument("--output-dir", default="output", help="Output directory for saved videos")

    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Config loaded from {args.config}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    else:
        logger.warning(f"Config file {args.config} not found, using default settings")

    # Set confidence threshold from args
    if not config.get('detection'):
        config['detection'] = {}
    config['detection']['confidence'] = args.conf_thres

    # Initialize tracker
    try:
        tracker = ObjectTracker(args.tracker, config)
        logger.info(f"Initialized {args.tracker} tracker")
    except Exception as e:
        logger.error(f"Error initializing tracker: {e}")
        tracker = None

    # Run detection and tracking
    try:
        run(
            model_path=args.model,
            source=args.source,
            tracker=tracker,
            config=config,
            logger=logger,
            display=args.display,
            save_video=args.save_video,
            output_dir=args.output_dir
        )
        logger.info("Detection and tracking completed successfully")
    except Exception as e:
        logger.error(f"Error in detection and tracking: {e}")
    finally:
        # Clean up resources
        finalize_visualizer()
        logger.info("Resources cleaned up")