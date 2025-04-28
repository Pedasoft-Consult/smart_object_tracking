#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Retraining Scheduler for Smart Object Tracking System.
Manages periodic model retraining based on feedback data.
"""

import time
import logging
import threading
import json
from pathlib import Path
import os


class RetrainingScheduler:
    """
    Schedules and manages model retraining based on collected feedback.
    Monitors feedback data and triggers retraining when sufficient data is available.
    """

    def __init__(self, feedback_manager, model_trainer, config):
        """
        Initialize retraining scheduler.

        Args:
            feedback_manager: Feedback manager instance for accessing feedback data
            model_trainer: Model trainer instance for training models
            config: Configuration dictionary
        """
        self.feedback_manager = feedback_manager
        self.model_trainer = model_trainer
        self.config = config
        self.logger = logging.getLogger('RetrainingScheduler')

        # Initialize state
        self.stop_event = threading.Event()
        self.scheduler_thread = None
        self.metadata_file = Path(
            self.config.get('model_registry', {}).get('directory', 'models')) / "retraining_scheduler.json"
        self._load_metadata()

        # Extract configuration
        self.check_interval = self.config.get('retraining', {}).get('check_interval', 3600)  # Default: 1 hour
        self.min_feedback_items = self.config.get('retraining', {}).get('min_feedback_items', 100)
        self.min_retraining_interval = self.config.get('retraining', {}).get('min_interval', 86400)  # Default: 1 day
        self.auto_deploy = self.config.get('retraining', {}).get('auto_deploy', True)

        self.logger.info(f"Retraining scheduler initialized with check interval {self.check_interval}s, "
                         f"minimum feedback items {self.min_feedback_items}")

    def _load_metadata(self):
        """Load scheduler metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}, creating new file")
                self._create_default_metadata()
        else:
            self._create_default_metadata()

    def _create_default_metadata(self):
        """Create default metadata structure"""
        self.metadata = {
            "last_check": 0,
            "last_retraining": 0,
            "retraining_history": [],
            "deployed_models": []
        }
        self._save_metadata()

    def _save_metadata(self):
        """Save scheduler metadata"""
        try:
            self.metadata["last_updated"] = time.time()

            # Create directory if it doesn't exist
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def start(self):
        """
        Start retraining scheduler thread.

        Returns:
            bool: True if started, False if already running
        """
        if self.scheduler_thread is not None and self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler thread is already running")
            return False

        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        self.logger.info("Retraining scheduler started")
        return True

    def stop(self):
        """
        Stop retraining scheduler thread.

        Returns:
            bool: True if stopped, False if not running
        """
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.logger.warning("Scheduler thread is not running")
            return False

        self.stop_event.set()
        self.scheduler_thread.join(timeout=10.0)

        self.logger.info("Retraining scheduler stopped")
        return True

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while not self.stop_event.is_set():
            try:
                # Check if it's time to check for retraining
                current_time = time.time()
                time_since_last_check = current_time - self.metadata["last_check"]

                if time_since_last_check >= self.check_interval:
                    # Update last check time
                    self.metadata["last_check"] = current_time
                    self._save_metadata()

                    # Check if retraining criteria are met
                    should_retrain = self._check_retraining_criteria()

                    if should_retrain:
                        # Trigger retraining
                        training_id = self._trigger_retraining()

                        if training_id:
                            self.logger.info(f"Retraining triggered with ID {training_id}")

                            # Record retraining event
                            self.metadata["last_retraining"] = current_time
                            self.metadata["retraining_history"].append({
                                "timestamp": current_time,
                                "training_id": training_id,
                                "triggered_by": "scheduler"
                            })
                            self._save_metadata()

                            # Wait for training to complete if auto-deploy is enabled
                            if self.auto_deploy:
                                self._wait_and_deploy(training_id)

                # Sleep until next interval check (but check stop event more frequently)
                for _ in range(min(int(self.check_interval / 10), 60)):  # Check at most every minute
                    if self.stop_event.is_set():
                        return
                    time.sleep(10)

            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

                # Sleep before retrying
                time.sleep(60)

    def _check_retraining_criteria(self):
        """
        Check if retraining criteria are met.

        Returns:
            bool: True if retraining should be triggered, False otherwise
        """
        # Check if enough time has passed since last retraining
        current_time = time.time()
        time_since_last_retraining = current_time - self.metadata["last_retraining"]

        if time_since_last_retraining < self.min_retraining_interval:
            self.logger.info(
                f"Not enough time since last retraining ({time_since_last_retraining:.2f}s / {self.min_retraining_interval}s)")
            return False

        # Check if enough feedback items are available
        feedback_stats = self.feedback_manager.get_statistics()
        processed_items = feedback_stats.get("processed_items", 0)

        if processed_items < self.min_feedback_items:
            self.logger.info(f"Not enough processed feedback items ({processed_items} / {self.min_feedback_items})")
            return False

        # Check if there's new feedback since last retraining
        new_feedback = False
        for item in self.feedback_manager.list_feedback_items(processed=True):
            if item.get("processing_time", 0) > self.metadata["last_retraining"]:
                new_feedback = True
                break

        if not new_feedback:
            self.logger.info("No new feedback since last retraining")
            return False

        # All criteria met
        self.logger.info("Retraining criteria met, triggering retraining")
        return True

    def _trigger_retraining(self):
        """
        Trigger model retraining.

        Returns:
            str: Training ID or None if failed
        """
        # Use feedback manager to trigger retraining
        return self.feedback_manager.trigger_retraining(
            model_trainer=self.model_trainer,
            min_feedback_items=self.min_feedback_items
        )

    def _wait_and_deploy(self, training_id):
        """
        Wait for training to complete and deploy the model.

        Args:
            training_id: Training ID to monitor
        """
        # Start monitoring in a separate thread to not block the scheduler
        threading.Thread(
            target=self._monitor_and_deploy_thread,
            args=(training_id,),
            daemon=True
        ).start()

    def _monitor_and_deploy_thread(self, training_id):
        """
        Thread for monitoring training progress and deploying the model.

        Args:
            training_id: Training ID to monitor
        """
        try:
            self.logger.info(f"Monitoring training progress for {training_id}")

            # Poll training status until completed or failed
            while True:
                # Check training status
                status = self.model_trainer.get_training_status(training_id)

                if status["status"] == "completed":
                    # Training completed successfully
                    self.logger.info(f"Training {training_id} completed successfully")

                    # Get model ID from training result
                    model_id = status.get("model_id")

                    if model_id:
                        # Deploy the model
                        self._deploy_model(model_id)

                    break

                elif status["status"] in ["failed", "stopped"]:
                    # Training failed or was stopped
                    self.logger.warning(f"Training {training_id} {status['status']}")
                    break

                # Sleep before checking again
                time.sleep(60)

        except Exception as e:
            self.logger.error(f"Error monitoring training {training_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _deploy_model(self, model_id):
        """
        Deploy a trained model.

        Args:
            model_id: Model ID to deploy

        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Deploying model {model_id}")

            # Get model details
            models = self.model_trainer.get_models()
            model_info = None

            for model in models:
                if model.get("model_id") == model_id:
                    model_info = model
                    break

            if model_info is None:
                self.logger.error(f"Model {model_id} not found in registry")
                return False

            # Check if model has been evaluated
            if "evaluation" not in model_info:
                # Evaluate model first
                self.logger.info(f"Evaluating model {model_id} before deployment")
                evaluation = self.model_trainer.evaluate(model_id=model_id)

                if "error" in evaluation:
                    self.logger.error(f"Model evaluation failed: {evaluation['error']}")
                    return False

            # Export model to deployment formats
            export_formats = self.config.get('model_registry', {}).get('export_formats', ['onnx'])

            exported_paths = {}
            for format in export_formats:
                try:
                    export_path = self.model_trainer.export_model(model_id=model_id, format=format)
                    if export_path:
                        exported_paths[format] = export_path
                        self.logger.info(f"Model exported to {format}: {export_path}")
                except Exception as e:
                    self.logger.error(f"Error exporting to {format}: {e}")

            # Update configuration to use new model
            # Get model registry path from model directory
            model_registry_dir = os.path.dirname(model_info.get("model_path", ""))

            # Use best format based on priority (ONNX preferred for edge deployment)
            deploy_path = None
            for format in ['onnx', 'tflite', 'pt']:
                if format in exported_paths:
                    deploy_path = exported_paths[format]
                    break

            if deploy_path is None:
                self.logger.error("No suitable export format found for deployment")
                return False

            # Get relative path for configuration
            if 'model_registry' in self.config and 'directory' in self.config['model_registry']:
                registry_base = self.config['model_registry']['directory']
                if deploy_path.startswith(registry_base):
                    deploy_path = os.path.relpath(deploy_path, registry_base)

            # Update configuration
            if 'models' not in self.config:
                self.config['models'] = {}

            if 'online_model' in self.config['models'] and deploy_path.endswith('.onnx'):
                # Use ONNX model as offline model
                self.config['models']['offline_model'] = deploy_path
                self.logger.info(f"Set offline_model to {deploy_path}")
            else:
                # Use as primary model
                self.config['models']['online_model'] = deploy_path
                self.logger.info(f"Set online_model to {deploy_path}")

            # Save updated configuration
            self._save_config()

            # Record deployment in metadata
            self.metadata["deployed_models"].append({
                "timestamp": time.time(),
                "model_id": model_id,
                "deploy_path": deploy_path,
                "metrics": model_info.get("evaluation", {}).get("metrics", {})
            })
            self._save_metadata()

            self.logger.info(f"Model {model_id} deployed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error deploying model {model_id}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _save_config(self):
        """Save updated configuration"""
        try:
            # Check if config file path is available
            config_path = self.config.get('config_path')

            if config_path and os.path.exists(config_path):
                import yaml
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
                self.logger.info(f"Configuration saved to {config_path}")
                return True
            else:
                self.logger.warning("Config path not available or doesn't exist")
                return False

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False

    def force_retrain(self):
        """
        Force immediate retraining regardless of criteria.

        Returns:
            str: Training ID or None if failed
        """
        self.logger.info("Forcing immediate retraining")

        # Trigger retraining directly
        training_id = self._trigger_retraining()

        if training_id:
            # Record forced retraining event
            self.metadata["last_retraining"] = time.time()
            self.metadata["retraining_history"].append({
                "timestamp": time.time(),
                "training_id": training_id,
                "triggered_by": "manual",
                "forced": True
            })
            self._save_metadata()

            # Monitor and deploy if auto-deploy is enabled
            if self.auto_deploy:
                self._wait_and_deploy(training_id)

        return training_id

    def get_status(self):
        """
        Get scheduler status.

        Returns:
            dict: Scheduler status
        """
        current_time = time.time()
        return {
            "is_running": self.scheduler_thread is not None and self.scheduler_thread.is_alive(),
            "last_check": self.metadata["last_check"],
            "time_since_check": current_time - self.metadata["last_check"],
            "last_retraining": self.metadata["last_retraining"],
            "time_since_retraining": current_time - self.metadata["last_retraining"],
            "retraining_count": len(self.metadata["retraining_history"]),
            "deployment_count": len(self.metadata["deployed_models"]),
            "check_interval": self.check_interval,
            "min_feedback_items": self.min_feedback_items,
            "min_retraining_interval": self.min_retraining_interval,
            "auto_deploy": self.auto_deploy
        }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    # Mock components for testing
    class MockFeedbackManager:
        def get_statistics(self):
            return {"processed_items": 150}

        def list_feedback_items(self, processed=None):
            return [{"processing_time": time.time()}]

        def trigger_retraining(self, model_trainer, min_feedback_items):
            return f"training_{int(time.time())}"


    class MockModelTrainer:
        def get_training_status(self, training_id):
            return {"status": "completed", "model_id": f"model_{training_id}"}

        def get_models(self):
            return [{"model_id": f"model_training_{int(time.time())}", "model_path": "/path/to/model.pt"}]

        def evaluate(self, model_id):
            return {"mAP_0.5": 0.85}

        def export_model(self, model_id, format):
            return f"/path/to/model.{format}"


    # Create scheduler
    config = {
        'retraining': {
            'check_interval': 60,  # Check every minute for testing
            'min_feedback_items': 50,
            'min_interval': 300,  # 5 minutes for testing
            'auto_deploy': True
        },
        'model_registry': {
            'directory': 'models',
            'export_formats': ['onnx', 'pt']
        },
        'config_path': 'configs/settings.yaml'
    }

    scheduler = RetrainingScheduler(MockFeedbackManager(), MockModelTrainer(), config)

    # Start scheduler
    scheduler.start()

    # Print status
    print(f"Scheduler status: {scheduler.get_status()}")

    # Simulate running for a while
    try:
        for _ in range(10):
            time.sleep(30)
            print(f"Scheduler status: {scheduler.get_status()}")
    except KeyboardInterrupt:
        pass

    # Force retrain
    training_id = scheduler.force_retrain()
    print(f"Forced retraining: {training_id}")

    # Stop scheduler
    scheduler.stop()