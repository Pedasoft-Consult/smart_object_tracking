#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shared test fixtures and configuration for pytest.
"""

import sys
import os
import pytest
import logging
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


def pytest_configure(config):
    """Configure pytest"""
    # Disable logging during tests
    logging.basicConfig(level=logging.ERROR)

    # Create logs directory if it doesn't exist
    logs_dir = Path(project_root) / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Set up test marker
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )


@pytest.fixture(scope="session")
def sample_frame():
    """Create a sample video frame for testing"""
    import numpy as np
    import cv2

    # Create a blank 720p frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Add some test objects
    # Person
    cv2.rectangle(frame, (100, 100), (200, 300), (0, 255, 0), 2)
    cv2.putText(frame, "Person", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Car
    cv2.rectangle(frame, (400, 200), (600, 300), (255, 0, 0), 2)
    cv2.putText(frame, "Car", (400, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Bicycle
    cv2.rectangle(frame, (700, 300), (800, 400), (0, 0, 255), 2)
    cv2.putText(frame, "Bicycle", (700, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame


@pytest.fixture(scope="session")
def sample_detections():
    """Create sample detection data for testing"""
    return [
        {'bbox': [100, 100, 200, 300], 'confidence': 0.92, 'class_id': 0, 'class_name': 'person'},
        {'bbox': [400, 200, 600, 300], 'confidence': 0.87, 'class_id': 2, 'class_name': 'car'},
        {'bbox': [700, 300, 800, 400], 'confidence': 0.81, 'class_id': 1, 'class_name': 'bicycle'}
    ]


@pytest.fixture(scope="session")
def sample_tracks():
    """Create sample track data for testing"""
    return [
        {'id': 1, 'bbox': [105, 105, 205, 305], 'class_id': 0, 'class_name': 'person', 'age': 5, 'hits': 5},
        {'id': 2, 'bbox': [405, 205, 605, 305], 'class_id': 2, 'class_name': 'car', 'age': 10, 'hits': 10},
        {'id': 3, 'bbox': [705, 305, 805, 405], 'class_id': 1, 'class_name': 'bicycle', 'age': 3, 'hits': 3}
    ]


@pytest.fixture
def mock_model():
    """Create a mock detection model for testing"""

    class MockModel:
        def __init__(self):
            self.names = {0: 'person', 1: 'bicycle', 2: 'car'}

        def __call__(self, image):
            # Return mock detection results
            class MockResults:
                def __init__(self):
                    self.xyxy = [
                        # Format: x1, y1, x2, y2, confidence, class
                        # Person
                        [100., 100., 200., 300., 0.92, 0.],
                        # Car
                        [400., 200., 600., 300., 0.87, 2.],
                        # Bicycle
                        [700., 300., 800., 400., 0.81, 1.]
                    ]

            return MockResults()

    return MockModel()


@pytest.fixture
def mock_preprocess():
    """Create a mock preprocess function for testing"""

    def preprocess(image):
        # Just return the image as-is for testing
        return image

    return preprocess