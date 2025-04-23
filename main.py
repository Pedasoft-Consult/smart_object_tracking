#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Smart Object Tracking System.
Handles switching between online/offline modes, loading appropriate models, and initiating tracking.
"""

import os
import sys
import argparse
import yaml
import logging
import time
from pathlib import Path
import cv2

# Modified path handling to prevent import conflicts
# Add project root to path but make it lower priority than system paths
project_root = str(Path(__file__).parent)
if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.append(project_root)

# Load configuration first before importing modules that might need it
def load_config():
    """
    Load configuration from settings file

    Returns:
        Configuration dictionary
    """
    config_path = Path(__file__).parent / "configs" / "settings.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}

# Load configuration early before other imports
config = load_config()

# Import local modules AFTER config is loaded
from tracker.tracker import ObjectTracker
# Import OTAUpdater directly from the module, avoiding __init__.py issue
from updater.ota_updater import OTAUpdater
from utils.connectivity import check_connectivity
from offline_queue import OfflineQueue
import detect_and_track


def setup_logging(config):
    """
    Set up logging

    Args:
        config: Configuration dictionary

    Returns:
        Logger instance
    """
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_dir = config.get('logging', {}).get('directory', 'logs')

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'tracking.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('ObjectTracking')


def check_for_updates(config, logger):
    """
    Check for model updates

    Args:
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        True if update was installed, False otherwise
    """
    if not config.get('updates', {}).get('enabled', False):
        logger.info("Updates are disabled")
        return False

    try:
        if not check_connectivity():
            logger.warning("No internet connection, skipping update check")
            return False

        updater = OTAUpdater(config)
        return updater.check_and_update()
    except Exception as e:
        logger.error(f"Error checking for updates: {e}")
        return False


def load_model(config, logger, online_mode=True):
    """
    Load the appropriate detection model

    Args:
        config: Configuration dictionary
        logger: Logger instance
        online_mode: Whether to use online or offline model

    Returns:
        Path to model file
    """
    model_dir = Path(config.get('models', {}).get('directory', 'models'))

    if online_mode:
        model_path = model_dir / config.get('models', {}).get('online_model', 'yolov5s.pt')
    else:
        model_path = model_dir / config.get('models', {}).get('offline_model', 'yolov5s-fp16.onnx')

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model: {model_path} ({'online' if online_mode else 'offline'} mode)")
    return str(model_path)


def start_tracking(config, logger, args):
    """
    Start the detection and tracking process

    Args:
        config: Configuration dictionary
        logger: Logger instance
        args: Command-line arguments
    """
    # Check network connectivity
    online_mode = check_connectivity()
    logger.info(f"Starting in {'online' if online_mode else 'offline'} mode")

    # Check for updates if in online mode
    if online_mode:
        update_installed = check_for_updates(config, logger)
        if update_installed:
            logger.info("Model updated successfully")

    # Load appropriate model
    try:
        model_path = load_model(config, logger, online_mode)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Initialize offline queue if in offline mode
    offline_queue = None
    if not online_mode:
        queue_dir = config.get('offline', {}).get('queue_directory', 'queue')
        offline_queue = OfflineQueue(queue_dir)
        logger.info(f"Initialized offline queue in {queue_dir}")

    # Start detection and tracking
    try:
        # Set source
        source = args.source
        if source is None:
            source = config.get('input', {}).get('default_source', 0)

        # Set tracker type
        tracker_type = args.tracker
        if tracker_type is None:
            tracker_type = config.get('tracking', {}).get('tracker', 'deep_sort')

        # Initialize tracker
        tracker = ObjectTracker(tracker_type, config)

        # Start detection and tracking loop
        logger.info(f"Starting detection and tracking from source: {source}")
        detect_and_track.run(
            model_path=model_path,
            source=source,
            tracker=tracker,
            config=config,
            offline_queue=offline_queue,
            online_mode=online_mode,
            logger=logger,
            display=args.display,
            save_video=args.save_video,
            output_dir=args.output_dir
        )

        logger.info("Detection and tracking completed")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error during tracking: {e}")
        logger.exception(e)
        return 1


def main():
    """Main entry point"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Smart Object Tracking System")
    parser.add_argument("--source", help="Source (0 for webcam, path for video file)")
    parser.add_argument("--display", action="store_true", help="Display output")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--tracker", choices=["deep_sort", "byte_track"], help="Tracker type")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Configuration is already loaded at the top of the file
    # but we'll reload it if a config file is specified via command line
    global config
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration from {args.config}: {e}")
            return 1

    # Set up logging
    logger = setup_logging(config)

    # Start tracking
    return start_tracking(config, logger, args)


if __name__ == "__main__":
    sys.exit(main())