import cv2
import time
import os
import numpy as np
import logging
from pathlib import Path


def debug_dump_tracking_state(error, tracks, detections, frame=None):
    """
    Save debug information when tracking errors occur

    Args:
        error: The error that occurred
        tracks: Current tracks
        detections: Current detections
        frame: Current frame (optional)
    """
    logger = logging.getLogger('TrackingDebug')

    try:
        # Create debug directory if it doesn't exist
        debug_dir = Path("debug_logs")
        debug_dir.mkdir(exist_ok=True)

        # Generate timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Save error details
        with open(debug_dir / f"error_{timestamp}.txt", "w") as f:
            f.write(f"Error: {error}\n\n")
            f.write(f"Tracks count: {len(tracks)}\n")
            f.write(f"Detections count: {len(detections)}\n\n")

            f.write("Track details:\n")
            for i, track in enumerate(tracks):
                f.write(f"Track {i}: {track}\n")

            f.write("\nDetection details:\n")
            for i, det in enumerate(detections):
                f.write(f"Detection {i}: type={type(det)}, value={det}\n")

        # Save frame if provided
        if frame is not None:
            cv2.imwrite(str(debug_dir / f"frame_{timestamp}.jpg"), frame)

        logger.info(f"Debug information saved to {debug_dir}")

    except Exception as e:
        logger.error(f"Error saving debug information: {e}")


def draw_detections(frame, detections):
    """
    Draw detection bounding boxes and labels on the frame

    Args:
        frame: The frame to draw on
        detections: List of detection dictionaries

    Returns:
        frame: The frame with detections drawn
    """
    try:
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        logger = logging.getLogger('TrackingDebug')
        logger.error(f"Error drawing detections: {e}")

    return frame


def draw_tracks(frame, tracks):
    """
    Draw tracking bounding boxes and IDs on the frame

    Args:
        frame: The frame to draw on
        tracks: List of tracks

    Returns:
        frame: The frame with tracks drawn
    """
    try:
        for track in tracks:
            if isinstance(track, dict):  # Handle dictionary format
                track_id = track.get('id', 0)
                x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif hasattr(track, 'track_id'):  # Handle Track object format with track_id attribute
                track_id = track.track_id
                # Check if bbox is a property or method
                if hasattr(track, 'bbox'):
                    if callable(track.bbox):
                        bbox = track.bbox()
                    else:
                        bbox = track.bbox
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Handle generic track objects with tlbr or tlwh format
            elif hasattr(track, 'tlbr'):
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = getattr(track, 'track_id', getattr(track, 'id', 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            elif hasattr(track, 'tlwh'):
                x, y, w, h = map(int, track.tlwh)
                track_id = getattr(track, 'track_id', getattr(track, 'id', 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Last resort - try to access as generic object
            else:
                # Get track_id or id attribute
                track_id = getattr(track, 'track_id', getattr(track, 'id', 0))

                # Try to get bounding box coordinates - check common attribute names
                bbox = None
                for attr_name in ['bbox', 'box', 'rect', 'position', 'pos']:
                    if hasattr(track, attr_name):
                        bbox_attr = getattr(track, attr_name)
                        if callable(bbox_attr):
                            bbox = bbox_attr()
                        else:
                            bbox = bbox_attr
                        break

                if bbox is not None:
                    # Handle different bbox formats (xyxy, xywh, etc.)
                    if len(bbox) == 4:
                        # Check if format is xywh
                        if all(isinstance(b, (int, float)) for b in bbox):
                            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:  # Width/height format
                                x1, y1, w, h = map(int, bbox)
                                x2, y2 = x1 + w, y1 + h
                            else:  # x1,y1,x2,y2 format
                                x1, y1, x2, y2 = map(int, bbox)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 2)
    except Exception as e:
        logger = logging.getLogger('TrackingDebug')
        logger.error(f"Error drawing tracks: {e}")

    return frame


# Integration with tracker_visualizer module
def draw_and_log_tracks(frame, tracks, model, frame_count):
    """
    Wrapper to draw and log tracks using the tracker_visualizer module.
    Falls back to internal implementation if the module fails.

    Args:
        frame: The frame to draw on
        tracks: The tracks to draw
        model: The detection model
        frame_count: Current frame number

    Returns:
        frame: The frame with visualizations
    """
    try:
        # Try to use the tracker_visualizer module
        from tracker_visualizer import draw_and_log_tracks as external_draw_and_log_tracks
        return external_draw_and_log_tracks(frame, tracks, model, frame_count)
    except Exception as e:
        logger = logging.getLogger('TrackingDebug')
        logger.error(f"Error using tracker_visualizer: {e}, falling back to internal implementation")
        # Fall back to internal implementation
        return draw_tracks(frame, tracks)


def finalize_visualizer():
    """
    Clean up resources from the visualizer module.

    Returns:
        bool: Success status
    """
    try:
        from tracker_visualizer import finalize_visualizer as external_finalize_visualizer
        external_finalize_visualizer()
        return True
    except Exception as e:
        logger = logging.getLogger('TrackingDebug')
        logger.error(f"Error finalizing visualizer: {e}")
        return False


def draw_fps(frame, fps):
    """
    Draw FPS indicator on frame

    Args:
        frame: The frame to draw on
        fps: Current FPS value

    Returns:
        frame: The frame with FPS drawn
    """
    try:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    except Exception as e:
        logger = logging.getLogger('TrackingDebug')
        logger.error(f"Error drawing FPS: {e}")

    return frame