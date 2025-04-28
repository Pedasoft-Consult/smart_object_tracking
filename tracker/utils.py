#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for object tracking.
Provides helper functions for feature extraction and cost calculation.
"""

import numpy as np
import cv2
import torch
import logging
from torchvision import transforms


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1: First box in format [x1, y1, x2, y2]
        box2: Second box in format [x1, y1, x2, y2]

    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def extract_features(frame, bbox, model):
    """
    Extract appearance features from detection

    Args:
        frame: Video frame
        bbox: Bounding box in format [x1, y1, x2, y2]
        model: Feature extraction model

    Returns:
        Feature vector
    """
    try:
        # Ensure bbox coordinates are integers and within frame bounds
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1], int(bbox[2]))
        y2 = min(frame.shape[0], int(bbox[3]))

        # Skip if box is too small
        if x2 - x1 < 10 or y2 - y1 < 10:
            return np.zeros(512)  # Return zero feature vector

        # Extract ROI
        roi = frame[y1:y2, x1:x2]

        # Preprocessing for ResNet
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Convert to RGB (OpenCV uses BGR)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Apply preprocessing
        input_tensor = preprocess(roi_rgb)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        # Move to GPU if available
        if torch.cuda.is_available():
            input_batch = input_batch.cuda()

        # Extract features
        with torch.no_grad():
            features = model(input_batch)

        # Convert to numpy array
        features_np = features.cpu().numpy().flatten()

        # Normalize
        features_norm = features_np / np.linalg.norm(features_np)

        return features_norm

    except Exception as e:
        logging.getLogger('ObjectTracking').warning(f"Feature extraction failed: {e}")
        return np.zeros(512)  # Return zero feature vector on error


def calculate_feature_distance(features1, features2):
    """
    Calculate distance between feature vectors

    Args:
        features1: First feature vector
        features2: Second feature vector

    Returns:
        Distance value (lower is more similar)
    """
    if features1 is None or features2 is None:
        return 1.0  # Maximum distance

    # Use cosine distance
    similarity = np.dot(features1, features2)

    # Bound similarity between 0 and 1
    similarity = max(0, min(similarity, 1.0))

    # Convert to distance (0=same, 1=different)
    return 1.0 - similarity


def create_cost_matrix(tracks, detections, detection_features, iou_threshold=0.3, max_feature_distance=0.5):
    """
    Create cost matrix for track-detection assignment

    Args:
        tracks: List of Track objects
        detections: List of detections [x1, y1, x2, y2, confidence, class_id]
        detection_features: List of feature vectors for detections
        iou_threshold: Minimum IoU for possible match
        max_feature_distance: Maximum feature distance for possible match

    Returns:
        Cost matrix
    """
    logger = logging.getLogger('ObjectTracking')

    num_tracks = len(tracks)
    num_detections = len(detections)

    # Initialize cost matrix with high values
    cost_matrix = np.ones((num_tracks, num_detections)) * 2.0  # Higher than possible IoU-based cost

    # Fill in cost matrix based on IoU and feature similarity
    for i, track in enumerate(tracks):
        track_bbox = track.bbox

        for j, detection in enumerate(detections):
            try:
                # Extract detection bounding box based on format
                if isinstance(detection, dict) and 'bbox' in detection:
                    det_bbox = detection['bbox']
                elif isinstance(detection, (list, tuple, np.ndarray)):
                    if len(detection) >= 4:
                        det_bbox = [detection[0], detection[1], detection[2], detection[3]]
                    else:
                        logger.warning(f"Detection {j} has insufficient elements: {detection}")
                        continue
                else:
                    logger.warning(f"Unsupported detection type for index {j}: {type(detection)}")
                    continue

                # Ensure bbox has 4 elements and is in the right format
                if not isinstance(det_bbox, (list, tuple, np.ndarray)) or len(det_bbox) != 4:
                    logger.warning(f"Invalid bbox format: {det_bbox}")
                    continue

                # Calculate IoU
                iou = calculate_iou(track_bbox, det_bbox)

                # Skip if IoU is below threshold
                if iou < iou_threshold:
                    continue

                # Set cost based on IoU
                iou_cost = 1.0 - iou

                # If features are available, incorporate feature similarity
                if detection_features and len(detection_features) > j and track.features:
                    # Use latest track features
                    track_features = track.features[-1]
                    detection_feature = detection_features[j]

                    # Calculate feature distance
                    feature_distance = calculate_feature_distance(track_features, detection_feature)

                    # Skip if feature distance is above threshold
                    if feature_distance > max_feature_distance:
                        continue

                    # Combine IoU and feature costs
                    # Give more weight to appearance for longer tracks
                    if track.hits > 5:
                        cost = 0.3 * iou_cost + 0.7 * feature_distance
                    else:
                        cost = 0.7 * iou_cost + 0.3 * feature_distance
                else:
                    # Use only IoU-based cost
                    cost = iou_cost

                # Set final cost
                cost_matrix[i, j] = cost

            except Exception as e:
                logger.error(f"Error calculating cost for track {i} and detection {j}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Keep default high cost for this pair

    return cost_matrix


def kalman_filter_predict(track):
    """
    Apply Kalman filter prediction step

    Args:
        track: Track object with Kalman filter state

    Returns:
        Predicted state
    """
    # Simple implementation - for full Kalman filter, consider using FilterPy library
    if not hasattr(track, 'kf_state'):
        # Initialize Kalman filter state
        # [x, y, width, height, dx, dy, dw, dh]
        track.kf_state = np.zeros(8)
        track.kf_state[0] = (track.bbox[0] + track.bbox[2]) / 2  # Center x
        track.kf_state[1] = (track.bbox[1] + track.bbox[3]) / 2  # Center y
        track.kf_state[2] = track.bbox[2] - track.bbox[0]  # Width
        track.kf_state[3] = track.bbox[3] - track.bbox[1]  # Height

        # Initial state covariance
        track.kf_covariance = np.eye(8) * 10

        # Process noise
        track.kf_process_noise = np.eye(8) * 0.01

        return track.kf_state

    # State transition matrix (constant velocity model)
    F = np.eye(8)
    F[0, 4] = 1.0  # x += dx
    F[1, 5] = 1.0  # y += dy
    F[2, 6] = 1.0  # width += dw
    F[3, 7] = 1.0  # height += dh

    # Predict state
    track.kf_state = F @ track.kf_state

    # Predict covariance
    track.kf_covariance = F @ track.kf_covariance @ F.T + track.kf_process_noise

    return track.kf_state


def kalman_filter_update(track, measurement):
    """
    Apply Kalman filter update step

    Args:
        track: Track object with Kalman filter state
        measurement: New measurement [x1, y1, x2, y2]

    Returns:
        Updated state
    """
    if not hasattr(track, 'kf_state'):
        return kalman_filter_predict(track)

    # Convert bbox to state vector format
    z = np.zeros(4)
    z[0] = (measurement[0] + measurement[2]) / 2  # Center x
    z[1] = (measurement[1] + measurement[3]) / 2  # Center y
    z[2] = measurement[2] - measurement[0]  # Width
    z[3] = measurement[3] - measurement[1]  # Height

    # Measurement matrix (maps state to measurement)
    H = np.zeros((4, 8))
    H[0, 0] = 1.0  # x
    H[1, 1] = 1.0  # y
    H[2, 2] = 1.0  # width
    H[3, 3] = 1.0  # height

    # Measurement noise
    R = np.eye(4) * 0.1

    # Kalman gain
    K = track.kf_covariance @ H.T @ np.linalg.inv(H @ track.kf_covariance @ H.T + R)

    # Update state
    track.kf_state = track.kf_state + K @ (z - H @ track.kf_state)

    # Update covariance
    track.kf_covariance = (np.eye(8) - K @ H) @ track.kf_covariance

    return track.kf_state


def state_to_bbox(state):
    """
    Convert Kalman filter state to bounding box

    Args:
        state: Kalman filter state [x, y, width, height, dx, dy, dw, dh]

    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    x, y, w, h = state[:4]
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]