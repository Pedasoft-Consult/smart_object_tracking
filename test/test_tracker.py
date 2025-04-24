#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the object tracker implementation.
"""

import sys
import os
import pytest
import numpy as np
import cv2
from pathlib import Path
import yaml

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules to test
from tracker.tracker import Track, DeepSORTTracker, ByteTracker, ObjectTracker


# Mock config for testing
@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary for testing"""
    return {
        "tracking": {
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
            "max_feature_distance": 0.5,
            "high_threshold": 0.6,
            "low_threshold": 0.1
        }
    }


# Test Track class
class TestTrackClass:
    def test_track_initialization(self):
        """Test that a track can be initialized correctly"""
        track_id = 1
        bbox = [100, 200, 150, 250]
        class_id = 0
        confidence = 0.9

        track = Track(track_id, bbox, class_id, confidence)

        assert track.track_id == track_id
        assert track.bbox == bbox
        assert track.class_id == class_id
        assert track.confidence == confidence
        assert track.age == 0
        assert track.time_since_update == 0
        assert track.hits == 1

        # Check center point
        expected_center = [(100 + 150) / 2, (200 + 250) / 2]
        assert track.trail[0] == expected_center

    def test_track_update(self):
        """Test that a track can be updated correctly"""
        track = Track(1, [100, 200, 150, 250], 0, 0.9)

        # Update with new bbox
        new_bbox = [110, 210, 160, 260]
        new_confidence = 0.95
        track.update(new_bbox, new_confidence)

        assert track.bbox == new_bbox
        assert track.confidence == new_confidence
        assert track.time_since_update == 0
        assert track.hits == 2
        assert track.age == 1

        # Check that trail was updated
        expected_center = [(110 + 160) / 2, (210 + 260) / 2]
        assert track.trail[1] == expected_center

    def test_track_predict(self):
        """Test track prediction based on motion"""
        track = Track(1, [100, 200, 150, 250], 0, 0.9)

        # First prediction should return the same bbox (not enough history)
        predicted_bbox = track.predict()
        assert predicted_bbox == track.bbox
        assert track.age == 1
        assert track.time_since_update == 1

        # Update track with new position
        track.update([110, 210, 160, 260])

        # Now predict based on motion
        predicted_bbox = track.predict()

        # Should predict linear motion: moved +10 in x and y
        expected_bbox = [120, 220, 170, 270]
        assert np.allclose(predicted_bbox, expected_bbox)
        assert track.age == 2
        assert track.time_since_update == 1


# Test DeepSORTTracker class
class TestDeepSORTTracker:
    def test_tracker_initialization(self, mock_config):
        """Test that the DeepSORT tracker can be initialized"""
        tracker = DeepSORTTracker(mock_config)

        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert tracker.max_feature_distance == 0.5
        assert tracker.tracks == []
        assert tracker.next_id == 1

    def test_tracker_update_with_no_detections(self, mock_config):
        """Test tracker update with no detections"""
        tracker = DeepSORTTracker(mock_config)

        result = tracker.update([], None)
        assert result == []

    def test_tracker_creates_new_tracks(self, mock_config):
        """Test that the tracker creates new tracks for detections"""
        tracker = DeepSORTTracker(mock_config)

        # Create some detections
        detections = [
            [100, 200, 150, 250, 0.9, 0],  # Person
            [300, 100, 400, 200, 0.8, 2],  # Car
        ]

        result = tracker.update(detections, None)

        # Should have created two tracks
        assert len(tracker.tracks) == 2

        # Check track properties
        assert tracker.tracks[0].track_id == 1
        assert tracker.tracks[0].bbox == [100, 200, 150, 250]
        assert tracker.tracks[0].class_id == 0
        assert tracker.tracks[0].confidence == 0.9

        assert tracker.tracks[1].track_id == 2
        assert tracker.tracks[1].bbox == [300, 100, 400, 200]
        assert tracker.tracks[1].class_id == 2
        assert tracker.tracks[1].confidence == 0.8

        # Next ID should be incremented
        assert tracker.next_id == 3

        # Active tracks should be empty since min_hits=3
        assert len(result) == 0

    def test_tracker_updates_existing_tracks(self, mock_config):
        """Test that the tracker updates existing tracks with new detections"""
        tracker = DeepSORTTracker(mock_config)

        # Create initial tracks
        detections1 = [
            [100, 200, 150, 250, 0.9, 0],  # Person
        ]
        tracker.update(detections1, None)

        # Check initial track
        assert len(tracker.tracks) == 1
        track_id = tracker.tracks[0].track_id
        assert track_id == 1
        assert tracker.tracks[0].bbox == [100, 200, 150, 250]
        assert tracker.tracks[0].hits == 1

        # Update with nearby detection
        detections2 = [
            [110, 210, 160, 260, 0.95, 0],  # Person moved slightly
        ]
        tracker.update(detections2, None)

        # Should have updated the existing track, not created a new one
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].track_id == track_id
        assert tracker.tracks[0].bbox == [110, 210, 160, 260]
        assert tracker.tracks[0].hits == 2
        assert tracker.tracks[0].confidence == 0.95

    def test_tracker_handles_missed_detections(self, mock_config):
        """Test that the tracker handles missed detections correctly"""
        tracker = DeepSORTTracker(mock_config)

        # Create initial track
        detections1 = [
            [100, 200, 150, 250, 0.9, 0],  # Person
        ]
        tracker.update(detections1, None)

        # Update with no detections
        tracker.update([], None)

        # Track should still exist but be marked as unmatched
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].time_since_update == 1

        # Update several more times with no detections
        for _ in range(mock_config["tracking"]["max_age"] - 1):
            tracker.update([], None)

        # Track should still exist at max_age
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].time_since_update == mock_config["tracking"]["max_age"]

        # One more update should remove the track
        tracker.update([], None)
        assert len(tracker.tracks) == 0

    def test_get_active_tracks(self, mock_config):
        """Test getting active tracks based on hits and age"""
        tracker = DeepSORTTracker(mock_config)

        # Create initial track
        detections = [
            [100, 200, 150, 250, 0.9, 0],  # Person
        ]

        # Update track min_hits times
        for i in range(mock_config["tracking"]["min_hits"]):
            # Move slightly each time
            bbox = [100 + i * 10, 200 + i * 10, 150 + i * 10, 250 + i * 10]
            detections = [[*bbox, 0.9, 0]]
            result = tracker.update(detections, None)

            # Track shouldn't be active until it reaches min_hits
            if i < mock_config["tracking"]["min_hits"] - 1:
                assert len(result) == 0
            else:
                assert len(result) == 1
                # Fix: Check the track_id attribute of the track object, not compare directly
                assert result[0].track_id == 1


# Test ByteTracker class
class TestByteTracker:
    def test_tracker_initialization(self, mock_config):
        """Test that the ByteTrack tracker can be initialized"""
        tracker = ByteTracker(mock_config)

        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert tracker.high_threshold == 0.6
        assert tracker.low_threshold == 0.1
        assert tracker.tracks == []
        assert tracker.next_id == 1

    def test_high_confidence_tracking(self, mock_config):
        """Test that high confidence detections are tracked"""
        tracker = ByteTracker(mock_config)

        # Create high confidence detection
        detections = [
            [100, 200, 150, 250, 0.9, 0],  # High confidence (above high_threshold)
        ]

        tracker.update(detections, None)

        # Should have created a track
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].track_id == 1
        assert tracker.tracks[0].bbox == [100, 200, 150, 250]

    def test_low_confidence_tracking(self, mock_config):
        """Test that low confidence detections are used for matching but not new tracks"""
        tracker = ByteTracker(mock_config)

        # First create a track with high confidence
        high_detections = [
            [100, 200, 150, 250, 0.9, 0],
        ]
        tracker.update(high_detections, None)

        # Then update with only low confidence detection nearby
        low_detections = [
            [110, 210, 160, 260, 0.2, 0],  # Low confidence but above low_threshold
        ]
        tracker.update(low_detections, None)

        # Should have updated existing track but not created new ones
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].track_id == 1
        assert tracker.tracks[0].bbox == [110, 210, 160, 260]

        # Now update with very low confidence detection (below low_threshold)
        very_low_detections = [
            [120, 220, 170, 270, 0.05, 0],  # Very low confidence (below low_threshold)
        ]
        tracker.update(very_low_detections, None)

        # Track should not be updated with very low confidence
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].bbox != [120, 220, 170, 270]
        assert tracker.tracks[0].time_since_update == 1


# Test ObjectTracker factory class
class TestObjectTracker:
    def test_creates_deep_sort_tracker(self, mock_config):
        """Test that ObjectTracker creates a DeepSORT tracker when requested"""
        tracker = ObjectTracker("deep_sort", mock_config)

        assert isinstance(tracker.tracker, DeepSORTTracker)
        assert tracker.tracker_type == "deep_sort"

    def test_creates_byte_track_tracker(self, mock_config):
        """Test that ObjectTracker creates a ByteTrack tracker when requested"""
        tracker = ObjectTracker("byte_track", mock_config)

        assert isinstance(tracker.tracker, ByteTracker)
        assert tracker.tracker_type == "byte_track"

    def test_fallback_to_deep_sort(self, mock_config):
        """Test that ObjectTracker defaults to DeepSORT for unknown tracker types"""
        tracker = ObjectTracker("unknown_tracker", mock_config)

        assert isinstance(tracker.tracker, DeepSORTTracker)


if __name__ == "__main__":
    pytest.main()