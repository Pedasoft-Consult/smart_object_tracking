#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main integration module for Smart Object Tracking System.
Integrates all components into a complete system with continuous learning capabilities.
"""

import os
import sys
import yaml
import logging
import argparse
import threading
from pathlib import Path
from api import initialize, app as api_app

# Ensure project root is in path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)


def setup_logging(config=None):
    """Set up logging configuration"""
    if config is None:
        config = {}

    log_level = config.get('logging', {}).get('level', 'INFO')
    log_dir = config.get('logging', {}).get('directory', 'logs')

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'system.log')),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger('MainIntegration')


def load_config(config_path='configs/settings.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Store config path for reference
        config['config_path'] = config_path

        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return {}


def initialize_components(config):
    """Initialize all components of the system"""
    logger = logging.getLogger('Initialization')
    components = {}

    # Import required modules
    try:
        # Initialize dataset manager
        logger.info("Initializing dataset manager")
        from dataset_manager import DatasetManager

        dataset_dir = config.get('dataset', {}).get('directory', 'dataset')
        class_file = config.get('dataset', {}).get('class_file')

        components['dataset_manager'] = DatasetManager(
            dataset_dir=dataset_dir,
            annotation_format=config.get('dataset', {}).get('format', 'yolo'),
            class_file=class_file
        )

        # Initialize model trainer
        logger.info("Initializing model trainer")
        from model_trainer import ModelTrainer

        model_dir = config.get('models', {}).get('directory', 'models')

        components['model_trainer'] = ModelTrainer(
            dataset_dir=dataset_dir,
            output_dir=model_dir,
            config=config
        )

        # Initialize feedback manager
        logger.info("Initializing feedback manager")
        from feedback_manager import FeedbackManager

        feedback_dir = config.get('feedback', {}).get('directory', 'feedback')

        components['feedback_manager'] = FeedbackManager(
            feedback_dir=feedback_dir,
            dataset_manager=components['dataset_manager']
        )

        # Initialize retraining scheduler
        if config.get('retraining', {}).get('enabled', True):
            logger.info("Initializing retraining scheduler")
            from retraining_scheduler import RetrainingScheduler

            components['retraining_scheduler'] = RetrainingScheduler(
                feedback_manager=components['feedback_manager'],
                model_trainer=components['model_trainer'],
                config=config
            )

            # Start scheduler if auto_start is enabled
            if config.get('retraining', {}).get('auto_start', True):
                components['retraining_scheduler'].start()
                logger.info("Retraining scheduler started")

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return components


def integrate_with_api(components):
    """Integrate components with API server"""
    logger = logging.getLogger('APIIntegration')

    try:
        # Import API module
        import api

        # Initialize annotation UI
        if 'feedback_manager' in components:
            from annotation_ui import init_annotation_ui
            init_annotation_ui(api.app, components['feedback_manager'])
            logger.info("Annotation UI initialized with API server")

        # Add components to API module for access
        api.components = components

        # Add API endpoint for feedback
        @api.app.route('/api/feedback/stats')
        def get_feedback_stats():
            """Get feedback statistics"""
            if 'feedback_manager' in components:
                stats = components['feedback_manager'].get_statistics()
                return api.jsonify(stats)
            else:
                return api.jsonify({"error": "Feedback manager not initialized"}), 404

        # Add API endpoint for training
        @api.app.route('/api/training/status')
        def get_training_status():
            """Get training status"""
            if 'model_trainer' in components:
                status = components['model_trainer'].get_training_status()
                return api.jsonify(status)
            else:
                return api.jsonify({"error": "Model trainer not initialized"}), 404

        # Add API endpoint for retraining
        @api.app.route('/api/retraining/status')
        def get_retraining_status():
            """Get retraining status"""
            if 'retraining_scheduler' in components:
                status = components['retraining_scheduler'].get_status()
                return api.jsonify(status)
            else:
                return api.jsonify({"error": "Retraining scheduler not initialized"}), 404

        # Add API endpoint to trigger retraining
        @api.app.route('/api/retraining/trigger', methods=['POST'])
        def trigger_retraining():
            """Trigger retraining"""
            if 'retraining_scheduler' in components:
                training_id = components['retraining_scheduler'].force_retrain()
                return api.jsonify({"success": True, "training_id": training_id})
            else:
                return api.jsonify({"error": "Retraining scheduler not initialized"}), 404

        logger.info("API endpoints integrated successfully")

    except Exception as e:
        logger.error(f"Error integrating with API: {e}")
        import traceback
        logger.error(traceback.format_exc())


# In the argument parser section (around line 45-50)
def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Smart Object Tracking System with Continuous Learning")
    parser.add_argument("--config", default="configs/settings.yaml", help="Path to configuration file")
    # Add the API argument
    parser.add_argument("--api", action="store_true", help="Start API server")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set up logging
    logger = setup_logging(config)
    logger.info("Starting Smart Object Tracking System with Continuous Learning")

    # Initialize components
    components = initialize_components(config)

    # Check if API server should be started
    if args.api:
        try:
            # Start API server with integrated components
            import api
            integrate_with_api(components)

            # Don't run detection and tracking to avoid conflicts
            api.app.run(
                host=config.get('api', {}).get('host', '0.0.0.0'),
                port=config.get('api', {}).get('port', 5000)
            )

        except ImportError:
            logger.error("API module not found. Cannot start API server.")
            return
    else:
        # Run main tracking system
        try:
            # Import main tracking module
            from main import start_tracking

            # Start tracking with integrated components
            tracking_system = start_tracking(config, logger, args)

        except ImportError:
            logger.error("Main tracking module not found. Cannot start tracking system.")


if __name__ == "__main__":
    main()