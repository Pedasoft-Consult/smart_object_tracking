#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance benchmarks for the object tracking system.
"""

import sys
import os
import pytest
import time
import numpy as np
import cv2
from pathlib import Path
import yaml

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules to test
from tracker.utils import calculate_iou
from tracker.tracker import DeepSORTTracker, ByteTracker
import detect_and_track


# Load sample configuration
@pytest.fixture
def test_config():
    """Load test configuration"""
    config_path = Path(project_root) / "configs" / "settings.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default test config
        config = {
            "device": "cpu",
            "detection": {
                "confidence": 0.25,
                "iou_threshold": 0.45,
                "frequency": 1
            },
            "tracking": {
                "tracker": "deep_sort",
                "max_age": 30,
                "min_hits": 3,
                "iou_threshold": 0.3,
                "max_feature_distance": 0.5
            }
        }
    return config


# Generate synthetic test images
@pytest.fixture
def synthetic_images():
    """Generate synthetic test images with objects"""
    images = []
    labels = []

    # Generate 10 test frames
    for i in range(10):
        # Create blank frame
        img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Add 5 objects that move slightly in each frame
        objects = []
        for j in range(5):
            # Calculate position with some motion
            x = 100 + j * 200 + i * 10
            y = 100 + j * 100 + (i % 3) * 5

            # Draw rectangle
            x1, y1, x2, y2 = x, y, x + 100, y + 100
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add to objects list
            objects.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': 0.9,
                'class_id': j % 3,
                'class_name': ['person', 'car', 'bicycle'][j % 3]
            })

        images.append(img)
        labels.append(objects)

    return images, labels


# Benchmark IoU calculation performance
class TestPerformanceIoU:
    @pytest.mark.benchmark(group="utils")
    def test_benchmark_iou(self, benchmark):
        """Benchmark IoU calculation performance"""
        # Generate 1000 random boxes
        boxes1 = np.random.rand(1000, 4) * 100
        boxes2 = boxes1 + np.random.rand(1000, 4) * 10 - 5  # Jitter around boxes1

        # Make sure x1 < x2 and y1 < y2
        for boxes in [boxes1, boxes2]:
            for i in range(len(boxes)):
                if boxes[i, 0] > boxes[i, 2]:
                    boxes[i, 0], boxes[i, 2] = boxes[i, 2], boxes[i, 0]
                if boxes[i, 1] > boxes[i, 3]:
                    boxes[i, 1], boxes[i, 3] = boxes[i, 3], boxes[i, 1]

        # Benchmark IoU calculation for 1000 box pairs
        def calc_iou_for_many():
            results = []
            for i in range(1000):
                iou = calculate_iou(boxes1[i], boxes2[i])
                results.append(iou)
            return results

        # Run benchmark
        result = benchmark(calc_iou_for_many)

        # Assert some basic properties about the result
        assert len(result) == 1000
        assert all(0 <= iou <= 1 for iou in result)


# Benchmark tracker performance
class TestTrackerPerformance:
    @pytest.mark.benchmark(group="tracker")
    def test_benchmark_deepsort(self, benchmark, test_config, synthetic_images):
        """Benchmark DeepSORT tracker performance"""
        images, labels = synthetic_images

        # Create tracker
        tracker = DeepSORTTracker(test_config)

        # Benchmark tracker update
        def run_tracker():
            for i in range(len(images)):
                tracker.update(labels[i], images[i])

        # Run benchmark
        benchmark(run_tracker)

    @pytest.mark.benchmark(group="tracker")
    def test_benchmark_bytetrack(self, benchmark, test_config, synthetic_images):
        """Benchmark ByteTrack tracker performance"""
        images, labels = synthetic_images

        # Create tracker
        tracker = ByteTracker(test_config)

        # Benchmark tracker update
        def run_tracker():
            for i in range(len(images)):
                tracker.update(labels[i], images[i])

        # Run benchmark
        benchmark(run_tracker)


# Benchmark detection frequency impact
class TestDetectionFrequency:
    def test_detection_frequency_impact(self, test_config):
        """Test impact of detection frequency on performance"""
        # This is more of an analysis than a unit test
        # In a real system, you would run this with different detection frequencies
        # and measure the performance/accuracy tradeoff

        frequencies = [1, 3, 5, 10]
        results = []

        for freq in frequencies:
            # Update config
            config = test_config.copy()
            config["detection"]["frequency"] = freq

            # Time how long it takes to process frames with this frequency
            start_time = time.time()

            # In a real test, you would process a video here
            # For this example, we'll just sleep proportionally
            time.sleep(0.1 / freq)

            elapsed = time.time() - start_time

            results.append({
                "frequency": freq,
                "elapsed_time": elapsed,
                "estimated_fps": 1.0 / elapsed if elapsed > 0 else 0
            })

        # Print results
        print("\nDetection Frequency Impact:")
        for result in results:
            print(f"Frequency {result['frequency']}: "
                  f"{result['elapsed_time']:.4f}s, "
                  f"Est. FPS: {result['estimated_fps']:.1f}")


# Integration benchmark - end-to-end pipeline
class TestIntegrationPerformance:
    def test_end_to_end_pipeline(self, test_config, synthetic_images):
        """Test end-to-end detection and tracking pipeline performance"""
        images, labels = synthetic_images

        # Create a mock model class for testing
        class MockModel:
            def __init__(self, labels):
                self.labels = labels
                self.names = {0: 'person', 1: 'car', 2: 'bicycle'}
                self.current_frame = 0

            def __call__(self, processed_frame):
                # Return mock results for the current frame
                class MockResults:
                    def __init__(self, detections):
                        self.xyxy = [np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3],
                             d['confidence'], d['class_id']]
                            for d in detections
                        ])]

                result = MockResults(self.labels[self.current_frame])
                self.current_frame = (self.current_frame + 1) % len(self.labels)
                return result

        # Create trackers
        deep_sort_tracker = DeepSORTTracker(test_config)
        byte_tracker = ByteTracker(test_config)

        # Benchmark different trackers
        trackers = {
            "DeepSORT": deep_sort_tracker,
            "ByteTrack": byte_tracker
        }

        results = {}

        for name, tracker in trackers.items():
            # Reset model
            model = MockModel(labels)

            # Measure time to process all frames
            start_time = time.time()

            for i in range(len(images)):
                # Process frame
                frame = images[i]

                # Run detection
                detection_results = model(frame)

                # Convert to format expected by tracker
                detections = []
                for det in detection_results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': model.names[int(cls)]
                    })

                # Run tracking
                tracks = tracker.update(detections, frame)

            elapsed = time.time() - start_time

            results[name] = {
                "elapsed_time": elapsed,
                "frames_per_second": len(images) / elapsed if elapsed > 0 else 0
            }

        # Print results
        print("\nEnd-to-End Pipeline Performance:")
        for name, result in results.items():
            print(f"{name}: {result['elapsed_time']:.4f}s, "
                  f"FPS: {result['frames_per_second']:.1f}")


if __name__ == "__main__":
    pytest.main()