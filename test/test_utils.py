#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the object tracking utility functions.
"""

import sys
import os
import pytest
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the modules to test
from tracker.utils import calculate_iou, calculate_feature_distance, create_cost_matrix, state_to_bbox


# Test cases for IoU calculation
class TestCalculateIoU:
    def test_perfect_overlap(self):
        """Test that boxes with perfect overlap have IoU of 1.0"""
        box1 = [10, 10, 20, 20]
        box2 = [10, 10, 20, 20]

        iou = calculate_iou(box1, box2)
        assert iou == 1.0

    def test_no_overlap(self):
        """Test that boxes with no overlap have IoU of 0.0"""
        box1 = [10, 10, 20, 20]
        box2 = [30, 30, 40, 40]

        iou = calculate_iou(box1, box2)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test that boxes with partial overlap have correct IoU"""
        # Box1: 10x10 area starting at (10,10)
        box1 = [10, 10, 20, 20]
        # Box2: 10x10 area starting at (15,15)
        box2 = [15, 15, 25, 25]

        # Overlap is 5x5 = 25
        # Total area is 10x10 + 10x10 - 5x5 = 175
        # IoU = 25/175 = 1/7 ≈ 0.143

        iou = calculate_iou(box1, box2)
        assert round(iou, 3) == 0.143

    def test_contained_box(self):
        """Test that a box contained within another has correct IoU"""
        # Box1: 20x20 area starting at (10,10)
        box1 = [10, 10, 30, 30]
        # Box2: 10x10 area starting at (15,15)
        box2 = [15, 15, 25, 25]

        # Overlap is 10x10 = 100
        # Total area is 20x20 + 10x10 - 10x10 = 400
        # IoU = 100/400 = 0.25

        iou = calculate_iou(box1, box2)
        assert iou == 0.25


# Test cases for feature distance calculation
class TestFeatureDistance:
    def test_identical_features(self):
        """Test that identical feature vectors have distance of 0.0"""
        features1 = np.array([0.1, 0.2, 0.3, 0.4])
        features2 = np.array([0.1, 0.2, 0.3, 0.4])

        distance = calculate_feature_distance(features1, features2)
        assert distance == 0.0

    def test_orthogonal_features(self):
        """Test that orthogonal feature vectors have distance of 1.0"""
        features1 = np.array([1, 0, 0, 0])
        features2 = np.array([0, 1, 0, 0])

        distance = calculate_feature_distance(features1, features2)
        assert distance == 1.0

    def test_partial_similarity(self):
        """Test that partially similar vectors have correct distance"""
        # Two vectors with cosine similarity of 0.5
        features1 = np.array([1, 1, 0, 0]) / np.sqrt(2)  # Normalized
        features2 = np.array([1, 0, 1, 0]) / np.sqrt(2)  # Normalized

        distance = calculate_feature_distance(features1, features2)
        assert round(distance, 1) == 0.5

    def test_None_features(self):
        """Test that None features result in maximum distance"""
        features1 = np.array([0.1, 0.2, 0.3, 0.4])
        features2 = None

        distance = calculate_feature_distance(features1, features2)
        assert distance == 1.0

        distance = calculate_feature_distance(None, features1)
        assert distance == 1.0


# Test cases for state to bbox conversion
class TestStateToBbox:
    def test_state_to_bbox_conversion(self):
        """Test conversion from Kalman filter state to bounding box"""
        # State: [center_x, center_y, width, height, dx, dy, dw, dh]
        state = np.array([100, 200, 50, 60, 5, 10, 0, 0])

        # Expected bbox: [x1, y1, x2, y2]
        expected_bbox = [75, 170, 125, 230]

        bbox = state_to_bbox(state)
        assert np.allclose(bbox, expected_bbox)