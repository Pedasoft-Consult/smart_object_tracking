#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object tracking implementation.
Supports DeepSORT and ByteTrack for tracking objects across video frames.
"""

import os
import numpy as np
import cv2
import logging
from pathlib import Path
import sys

# Add tracker utils module to path
sys.path.append(str(Path(__file__).parent))
from tracker.utils import calculate_iou, extract_features, create_cost_matrix


class Track:
    """Represents a single tracked object"""

    def __init__(self, track_id, bbox, class_id, confidence=None):
        """
        Initialize a new track

        Args:
            track_id: Unique ID for this track
            bbox: Bounding box in format [x1, y1, x2, y2]
            class_id: Object class ID
            confidence: Detection confidence
        """
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.age = 0  # How many frames this track has existed
        self.time_since_update = 0  # Frames since last update
        self.hits = 1  # Number of detections that have been assigned to this track
        self.trail = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]  # Center point trail
        self.features = []  # Feature vector history for appearance matching

    def update(self, bbox, confidence=None):
        """
        Update track with new detection

        Args:
            bbox: New bounding box
            confidence: New confidence score
        """
        self.bbox = bbox
        if confidence is not None:
            self.confidence = confidence
        self.time_since_update = 0
        self.hits += 1
        self.age += 1

        # Update center point trail
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.trail.append((center_x, center_y))

        # Limit trail length to avoid memory issues
        if len(self.trail) > 30:
            self.trail = self.trail[-30:]

    def predict(self):
        """
        Predict new location based on simple linear motion model

        Returns:
            Predicted bounding box
        """
        if len(self.trail) < 2:
            # Not enough history for prediction
            self.age += 1
            self.time_since_update += 1
            return self.bbox

        # Calculate velocity from last two positions
        last_center = self.trail[-1]
        prev_center = self.trail[-2]

        dx = last_center[0] - prev_center[0]
        dy = last_center[1] - prev_center[1]

        # Predict new center
        new_center_x = last_center[0] + dx
        new_center_y = last_center[1] + dy

        # Calculate width and height
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]

        # Update bounding box
        new_bbox = [
            new_center_x - width / 2,
            new_center_y - height / 2,
            new_center_x + width / 2,
            new_center_y + height / 2
        ]

        self.age += 1
        self.time_since_update += 1

        return new_bbox

    def add_feature(self, feature):
        """
        Add appearance feature

        Args:
            feature: Feature vector
        """
        self.features.append(feature)
        if len(self.features) > 10:
            self.features = self.features[-10:]


class DeepSORTTracker:
    """DeepSORT tracking algorithm implementation"""

    def __init__(self, config):
        """
        Initialize DeepSORT tracker

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tracks = []
        self.next_id = 1
        self.max_age = config.get('tracking', {}).get('max_age', 30)
        self.min_hits = config.get('tracking', {}).get('min_hits', 3)
        self.iou_threshold = config.get('tracking', {}).get('iou_threshold', 0.3)
        self.max_feature_distance = config.get('tracking', {}).get('max_feature_distance', 0.5)
        self.logger = logging.getLogger('DeepSORTTracker')

        # Load feature extractor model if available
        model_path = config.get('tracking', {}).get('feature_model', None)
        if model_path and os.path.exists(model_path):
            try:
                import torch
                self.feature_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
                self.feature_model.eval()
                if torch.cuda.is_available():
                    self.feature_model.cuda()
                self.logger.info(f"Loaded feature extraction model: {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load feature model: {e}")
                self.feature_model = None
        else:
            self.feature_model = None

    def update(self, detections, frame):
        """
        Update tracker with new detections

        Args:
            detections: List of detections [x1, y1, x2, y2, confidence, class_id]
            frame: Current video frame

        Returns:
            List of active tracks
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # If no detections, just return current tracks
        if detections is None or len(detections) == 0:
            return self.get_active_tracks()

        # Extract features from detection regions if feature model is available
        detection_features = []
        if self.feature_model is not None and frame is not None:
            for det in detections:
                bbox = det[:4].astype(int)
                feature = extract_features(frame, bbox, self.feature_model)
                detection_features.append(feature)

        # Match detections to existing tracks
        track_indices = list(range(len(self.tracks)))
        detection_indices = list(range(len(detections)))

        if len(track_indices) == 0:
            # If no existing tracks, create new tracks for all detections
            for i, det in enumerate(detections):
                self._initiate_track(det, detection_features[i] if detection_features else None)
            return self.get_active_tracks()

        # Create cost matrix for matching
        cost_matrix = create_cost_matrix(
            self.tracks, detections, detection_features,
            self.iou_threshold, self.max_feature_distance
        )

        # Use Hungarian algorithm for assignment
        from scipy.optimize import linear_sum_assignment
        track_indices_l, detection_indices_l = linear_sum_assignment(cost_matrix)

        # Update matched tracks
        for track_idx, det_idx in zip(track_indices_l, detection_indices_l):
            if cost_matrix[track_idx, det_idx] > 0.75:  # High cost means poor match
                # Mark as unmatched
                if track_idx in track_indices:
                    track_indices.remove(track_idx)
                if det_idx in detection_indices:
                    detection_indices.remove(det_idx)
                continue

            self.tracks[track_idx].update(
                detections[det_idx][:4],
                detections[det_idx][4] if len(detections[det_idx]) > 4 else None
            )

            # Add feature if available
            if detection_features and self.feature_model is not None:
                self.tracks[track_idx].add_feature(detection_features[det_idx])

            # Remove from unmatched lists
            if track_idx in track_indices:
                track_indices.remove(track_idx)
            if det_idx in detection_indices:
                detection_indices.remove(det_idx)

        # Create new tracks for unmatched detections
        for det_idx in detection_indices:
            self._initiate_track(
                detections[det_idx],
                detection_features[det_idx] if detection_features else None
            )

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self.get_active_tracks()

    def _initiate_track(self, detection, feature=None):
        """
        Create a new track from detection

        Args:
            detection: Detection [x1, y1, x2, y2, confidence, class_id]
            feature: Optional feature vector
        """
        bbox = detection[:4]
        confidence = detection[4] if len(detection) > 4 else None
        class_id = int(detection[5]) if len(detection) > 5 else 0

        # Create new track
        track = Track(self.next_id, bbox, class_id, confidence)

        # Add feature if available
        if feature is not None:
            track.add_feature(feature)

        # Add to tracks list
        self.tracks.append(track)

        # Increment next ID
        self.next_id += 1

    def get_active_tracks(self, min_hits=None, max_age=None):
        """
        Get list of active tracks

        Args:
            min_hits: Minimum number of detections to be considered active
            max_age: Maximum frames since last update to be considered active

        Returns:
            List of active Track objects
        """
        if min_hits is None:
            min_hits = self.min_hits
        if max_age is None:
            max_age = self.max_age

        return [t for t in self.tracks
                if t.hits >= min_hits and t.time_since_update <= max_age]


class ByteTracker:
    """ByteTrack tracking algorithm implementation"""

    def __init__(self, config):
        """
        Initialize ByteTrack tracker

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tracks = []
        self.next_id = 1
        self.max_age = config.get('tracking', {}).get('max_age', 30)
        self.min_hits = config.get('tracking', {}).get('min_hits', 3)
        self.iou_threshold = config.get('tracking', {}).get('iou_threshold', 0.3)
        self.logger = logging.getLogger('ByteTracker')

        # ByteTrack-specific parameters
        self.high_threshold = config.get('tracking', {}).get('high_threshold', 0.6)
        self.low_threshold = config.get('tracking', {}).get('low_threshold', 0.1)

    def update(self, detections, frame):
        """
        Update tracker with new detections

        Args:
            detections: List of detections [x1, y1, x2, y2, confidence, class_id]
            frame: Current video frame

        Returns:
            List of active tracks
        """
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # If no detections, just return current tracks
        if detections is None or len(detections) == 0:
            return self.get_active_tracks()

        # Split detections into high and low confidence
        high_score_detections = []
        low_score_detections = []

        for det in detections:
            if len(det) > 4 and det[4] >= self.high_threshold:
                high_score_detections.append(det)
            elif len(det) > 4 and det[4] >= self.low_threshold:
                low_score_detections.append(det)

        # First association: match tracks with high-score detections
        track_indices = list(range(len(self.tracks)))
        high_det_indices = list(range(len(high_score_detections)))

        if len(track_indices) > 0 and len(high_det_indices) > 0:
            # Create cost matrix based on IoU
            cost_matrix = np.zeros((len(track_indices), len(high_det_indices)))

            for i, track_idx in enumerate(track_indices):
                for j, det_idx in enumerate(high_det_indices):
                    track_bbox = self.tracks[track_idx].bbox
                    det_bbox = high_score_detections[det_idx][:4]
                    iou = calculate_iou(track_bbox, det_bbox)
                    cost_matrix[i, j] = 1 - iou  # Convert IoU to cost (lower is better)

            # Use Hungarian algorithm for assignment
            from scipy.optimize import linear_sum_assignment
            track_indices_l, det_indices_l = linear_sum_assignment(cost_matrix)

            # Update matched tracks
            for track_idx, det_idx in zip(track_indices_l, det_indices_l):
                if cost_matrix[track_idx, det_idx] > 1 - self.iou_threshold:
                    # Low IoU means poor match
                    continue

                self.tracks[track_idx].update(
                    high_score_detections[det_idx][:4],
                    high_score_detections[det_idx][4] if len(high_score_detections[det_idx]) > 4 else None
                )

                # Remove from unmatched lists
                if track_idx in track_indices:
                    track_indices.remove(track_idx)
                if det_idx in high_det_indices:
                    high_det_indices.remove(det_idx)

        # Second association: match remaining tracks with low-score detections
        if len(track_indices) > 0 and len(low_score_detections) > 0:
            low_det_indices = list(range(len(low_score_detections)))

            # Create cost matrix based on IoU
            cost_matrix = np.zeros((len(track_indices), len(low_det_indices)))

            for i, track_idx in enumerate(track_indices):
                for j, det_idx in enumerate(low_det_indices):
                    track_bbox = self.tracks[track_idx].bbox
                    det_bbox = low_score_detections[det_idx][:4]
                    iou = calculate_iou(track_bbox, det_bbox)
                    cost_matrix[i, j] = 1 - iou  # Convert IoU to cost

            # Use Hungarian algorithm for assignment
            from scipy.optimize import linear_sum_assignment
            track_indices_l, det_indices_l = linear_sum_assignment(cost_matrix)

            # Update matched tracks
            for track_idx, det_idx in zip(track_indices_l, det_indices_l):
                if cost_matrix[track_idx, det_idx] > 1 - self.iou_threshold:
                    continue

                self.tracks[track_idx].update(
                    low_score_detections[det_idx][:4],
                    low_score_detections[det_idx][4] if len(low_score_detections[det_idx]) > 4 else None
                )

                # Remove from unmatched track list
                if track_idx in track_indices:
                    track_indices.remove(track_idx)

        # Create new tracks for unmatched high-score detections
        for det_idx in high_det_indices:
            self._initiate_track(high_score_detections[det_idx])

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return self.get_active_tracks()

    def _initiate_track(self, detection):
        """
        Create a new track from detection

        Args:
            detection: Detection [x1, y1, x2, y2, confidence, class_id]
        """
        bbox = detection[:4]
        confidence = detection[4] if len(detection) > 4 else None
        class_id = int(detection[5]) if len(detection) > 5 else 0

        # Create new track
        track = Track(self.next_id, bbox, class_id, confidence)

        # Add to tracks list
        self.tracks.append(track)

        # Increment next ID
        self.next_id += 1

    def get_active_tracks(self, min_hits=None, max_age=None):
        """
        Get list of active tracks

        Args:
            min_hits: Minimum number of detections to be considered active
            max_age: Maximum frames since last update to be considered active

        Returns:
            List of active Track objects
        """
        if min_hits is None:
            min_hits = self.min_hits
        if max_age is None:
            max_age = self.max_age

        return [t for t in self.tracks
                if t.hits >= min_hits and t.time_since_update <= max_age]


class ObjectTracker:
    """Factory class for object trackers"""

    def __init__(self, tracker_type, config):
        """
        Initialize appropriate tracker

        Args:
            tracker_type: Type of tracker ('deep_sort' or 'byte_track')
            config: Configuration dictionary
        """
        self.logger = logging.getLogger('ObjectTracker')

        if tracker_type.lower() == 'deep_sort':
            self.logger.info("Initializing DeepSORT tracker")
            self.tracker = DeepSORTTracker(config)
        elif tracker_type.lower() == 'byte_track':
            self.logger.info("Initializing ByteTrack tracker")
            self.tracker = ByteTracker(config)
        else:
            self.logger.warning(f"Unknown tracker type: {tracker_type}, using DeepSORT")
            self.tracker = DeepSORTTracker(config)

        self.tracker_type = tracker_type

    def update(self, detections, frame):
        """
        Update tracker with new detections

        Args:
            detections: List of detections
            frame: Current video frame

        Returns:
            List of active tracks
        """
        return self.tracker.update(detections, frame)

    def get_active_tracks(self, min_hits=None, max_age=None):
        """
        Get list of active tracks

        Args:
            min_hits: Minimum number of detections to be considered active
            max_age: Maximum frames since last update to be considered active

        Returns:
            List of active Track objects
        """
        return self.tracker.get_active_tracks(min_hits, max_age)