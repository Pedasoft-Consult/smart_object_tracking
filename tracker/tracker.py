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

sys.path.append(str(Path(__file__).parent))
from tracker.utils import calculate_iou, extract_features, create_cost_matrix


class Track:
    def __init__(self, track_id, bbox, class_id, confidence=None):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.trail = [(center_x, center_y)]
        self.features = []

    def update(self, bbox, confidence=None):
        self.bbox = bbox
        if confidence is not None:
            self.confidence = confidence
        self.time_since_update = 0
        self.hits += 1
        self.age += 1
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.trail.append((center_x, center_y))
        if len(self.trail) > 30:
            self.trail = self.trail[-30:]

    def predict(self):
        if len(self.trail) < 2:
            self.age += 1
            self.time_since_update += 1
            return self.bbox
        last_center = self.trail[-1]
        prev_center = self.trail[-2]
        dx = last_center[0] - prev_center[0]
        dy = last_center[1] - prev_center[1]
        new_center_x = last_center[0] + dx
        new_center_y = last_center[1] + dy
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        new_bbox = [new_center_x - width / 2, new_center_y - height / 2,
                    new_center_x + width / 2, new_center_y + height / 2]
        self.age += 1
        self.time_since_update += 1
        return new_bbox

    def add_feature(self, feature):
        self.features.append(feature)
        if len(self.features) > 10:
            self.features = self.features[-10:]


class DeepSORTTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.next_id = 1
        self.max_age = config.get('tracking', {}).get('max_age', 30)
        self.min_hits = config.get('tracking', {}).get('min_hits', 3)
        self.iou_threshold = config.get('tracking', {}).get('iou_threshold', 0.3)
        self.max_feature_distance = config.get('tracking', {}).get('max_feature_distance', 0.5)
        self.high_threshold = config.get('tracking', {}).get('high_threshold', 0.6)
        self.low_threshold = config.get('tracking', {}).get('low_threshold', 0.1)
        self.logger = logging.getLogger('DeepSORTTracker')

    def update(self, detections, frame):
        """Update tracks with new detections"""
        # Update existing tracks with motion model
        for track in self.tracks:
            track.predict()

        # Handle case when no detections are present
        if detections is None or len(detections) == 0:
            return self.get_active_tracks()

        # Process detections to ensure consistent format
        processed_detections = []
        for det in detections:
            try:
                if isinstance(det, dict) and 'bbox' in det:
                    # Already in correct format
                    processed_detections.append(det)
                elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 4:
                    # Convert array-like to dictionary format
                    confidence = det[4] if len(det) > 4 else 0.0
                    class_id = int(det[5]) if len(det) > 5 else 0
                    processed_detections.append({
                        'bbox': det[:4],
                        'confidence': confidence,
                        'class_id': class_id
                    })
                else:
                    logging.getLogger('ByteTracker').warning(f"Skipping detection with unsupported format: {type(det)}")
            except Exception as e:
                logging.getLogger('ByteTracker').error(f"Error processing detection: {e}")

        # If we have no valid detections after processing, return
        if len(processed_detections) == 0:
            return self.get_active_tracks()

        # Now proceed with tracking using the standardized detections
        detection_features = []  # We're not using features here

        try:
            # Create cost matrix with standardized detections
            cost_matrix = create_cost_matrix(self.tracks, processed_detections, detection_features, self.iou_threshold,
                                             0.5)

            # Rest of the tracking logic with assignments
            from scipy.optimize import linear_sum_assignment
            track_indices_l, det_indices_l = linear_sum_assignment(cost_matrix)

            # Process the assignments
            matched_tracks = set()
            matched_dets = set()
            for track_idx, det_idx in zip(track_indices_l, det_indices_l):
                if cost_matrix[track_idx, det_idx] > 0.75:  # Cost threshold
                    continue

                det = processed_detections[det_idx]
                bbox = det['bbox']
                confidence = det.get('confidence')

                # Update the track
                self.tracks[track_idx].update(bbox, confidence)
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)

            # Create new tracks for unmatched detections
            for i, det in enumerate(processed_detections):
                if i not in matched_dets:
                    self._initiate_track(det)

            # Remove old tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

            return self.get_active_tracks()

        except Exception as e:
            logger = logging.getLogger('ByteTracker')
            logger.error(f"Error in tracking update: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return existing tracks as fallback
            return self.get_active_tracks()

    def _match_tracks(self, detections):
        if len(detections) == 0:
            return
        unmatched_tracks = list(range(len(self.tracks)))
        det_indices = list(range(len(detections)))
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                bbox = det['bbox'] if isinstance(det, dict) else det[:4]
                iou = calculate_iou(track.bbox, bbox)
                cost_matrix[i, j] = 1 - iou
        from scipy.optimize import linear_sum_assignment
        track_indices_l, det_indices_l = linear_sum_assignment(cost_matrix)
        for track_idx, det_idx in zip(track_indices_l, det_indices_l):
            if cost_matrix[track_idx, det_idx] > 1 - self.iou_threshold:
                continue
            det = detections[det_idx]
            bbox = det['bbox'] if isinstance(det, dict) else det[:4]
            confidence = det.get('confidence') if isinstance(det, dict) else det[4] if len(det) > 4 else None
            self.tracks[track_idx].update(bbox, confidence)
            if track_idx in unmatched_tracks:
                unmatched_tracks.remove(track_idx)
            if det_idx in det_indices:
                det_indices.remove(det_idx)
        for det_idx in det_indices:
            self._initiate_track(detections[det_idx])

    def _initiate_track(self, detection):
        bbox = detection['bbox'] if isinstance(detection, dict) else detection[:4]
        confidence = detection.get('confidence') if isinstance(detection, dict) else detection[4] if len(detection) > 4 else None
        class_id = detection.get('class_id', 0) if isinstance(detection, dict) else int(detection[5]) if len(detection) > 5 else 0
        track = Track(self.next_id, bbox, class_id, confidence)
        self.tracks.append(track)
        self.next_id += 1

    def get_active_tracks(self, min_hits=None, max_age=None):
        if min_hits is None:
            min_hits = self.min_hits
        if max_age is None:
            max_age = self.max_age
        return [t for t in self.tracks if t.hits >= min_hits and t.time_since_update <= max_age]


class ByteTracker:
    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.next_id = 1
        self.max_age = config.get('tracking', {}).get('max_age', 30)
        self.min_hits = config.get('tracking', {}).get('min_hits', 3)
        self.iou_threshold = config.get('tracking', {}).get('iou_threshold', 0.3)
        self.logger = logging.getLogger('ByteTracker')
        self.high_threshold = config.get('tracking', {}).get('high_threshold', 0.6)
        self.low_threshold = config.get('tracking', {}).get('low_threshold', 0.1)

    def update(self, detections, frame):
        for track in self.tracks:
            track.predict()
        if detections is None or len(detections) == 0:
            return self.get_active_tracks()
        detection_features = []
        cost_matrix = create_cost_matrix(self.tracks, detections, detection_features, self.iou_threshold, 0.5)
        from scipy.optimize import linear_sum_assignment
        track_indices_l, det_indices_l = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_dets = set()
        for track_idx, det_idx in zip(track_indices_l, det_indices_l):
            if cost_matrix[track_idx, det_idx] > 0.75:
                continue
            det = detections[det_idx]
            bbox = det['bbox'] if isinstance(det, dict) else det[:4]
            confidence = det.get('confidence') if isinstance(det, dict) else det[4] if len(det) > 4 else None
            self.tracks[track_idx].update(bbox, confidence)
            matched_tracks.add(track_idx)
            matched_dets.add(det_idx)
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self._initiate_track(det)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return self.get_active_tracks()

    def _initiate_track(self, detection):
        bbox = detection['bbox'] if isinstance(detection, dict) else detection[:4]
        confidence = detection.get('confidence') if isinstance(detection, dict) else detection[4] if len(detection) > 4 else None
        class_id = detection.get('class_id', 0) if isinstance(detection, dict) else int(detection[5]) if len(detection) > 5 else 0
        track = Track(self.next_id, bbox, class_id, confidence)
        self.tracks.append(track)
        self.next_id += 1

    def get_active_tracks(self, min_hits=None, max_age=None):
        if min_hits is None:
            min_hits = self.min_hits
        if max_age is None:
            max_age = self.max_age
        return [t for t in self.tracks if t.hits >= min_hits and t.time_since_update <= max_age]


class ObjectTracker:
    def __init__(self, tracker_type, config):
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
        return self.tracker.update(detections, frame)

    def get_active_tracks(self, min_hits=None, max_age=None):
        return self.tracker.get_active_tracks(min_hits, max_age)
