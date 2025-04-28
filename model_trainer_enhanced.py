#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Model Trainer for Smart Object Tracking System.
Core module for training object detection models with improvements for small datasets.
"""

import os
import json
import time
import logging
import threading
import subprocess
import sys
import shutil
from pathlib import Path
import torch
import glob
import matplotlib.pyplot as plt
import pandas as pd

# Import training status management methods
from training_status_updates import update_training_status, update_training_progress


class ModelTrainer:
    """Core trainer for object detection models"""

    def __init__(self, dataset_dir, output_dir, config):
        """
        Initialize model trainer.

        Args:
            dataset_dir: Path to dataset directory
            output_dir: Path to output directory for trained models
            config: Configuration dictionary
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logging.getLogger('ModelTrainer')

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self.metadata_file = self.output_dir / "training_metadata.json"
        self._load_metadata()

        # Initialize helper classes - will be created after init to avoid circular imports
        self.exporter = None
        self.evaluator = None

        # Store current training process
        self.current_process = None
        self.is_training = False
        self.stop_event = threading.Event()
        self.training_thread = None
        self.current_training_id = None

        self.logger.info(
            f"Model trainer initialized with dataset at {self.dataset_dir} and output at {self.output_dir}")

        # Initialize helpers after we're fully set up
        self._init_helpers()

    def _generate_training_graphs(self, training_dir: Path):
        results_csv = training_dir / "results.csv"
        if not results_csv.exists():
            self.logger.warning(f"Results CSV not found at {results_csv}")
            return

        df = pd.read_csv(results_csv)

        epochs = df.index
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Safe plotting for training losses
        loss_cols = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'box_loss', 'obj_loss', 'cls_loss']
        for col in loss_cols:
            if col in df.columns:
                axs[0].plot(epochs, df[col], label=col)
        axs[0].set_title('Training Losses')
        axs[0].legend()

        # Safe plotting for validation metrics
        map_cols = ['metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'mAP_0.5', 'mAP_0.5:0.95']
        for col in map_cols:
            if col in df.columns:
                axs[1].plot(epochs, df[col], label=col)
        axs[1].set_title('Validation mAP')
        axs[1].legend()

        plt.tight_layout()
        graphs_path = training_dir / "training_curves.png"
        plt.savefig(graphs_path)
        self.logger.info(f"Training curves saved to {graphs_path}")

    def _init_helpers(self):
        """Initialize helper components after self is initialized"""
        try:
            from model_exporter import ModelExporter
            from model_evaluator import ModelEvaluator

            self.exporter = ModelExporter(self.output_dir, self.config, self.metadata)
            self.evaluator = ModelEvaluator(self.output_dir, self.dataset_dir, self.config, self.metadata)
            self.logger.info("Initialized exporter and evaluator components")
        except ImportError as e:
            self.logger.warning(f"Could not import helper modules: {e}")
            self.logger.warning("Some functionality may be limited")

    def _load_metadata(self):
        """Load training metadata"""
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
            "models": [],
            "training_runs": [],
            "last_updated": time.time()
        }
        self._save_metadata()

    def _save_metadata(self):
        """Save training metadata"""
        try:
            self.metadata["last_updated"] = time.time()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    # Add wrapper methods to use the imported functions
    def _update_training_status(self, training_id, status, additional_info=None):
        """Wrapper for external training status update function"""
        return update_training_status(self.metadata_file, training_id, status, additional_info)

    def _update_training_progress(self, training_id, progress, metrics=None):
        """Wrapper for external training progress update function"""
        return update_training_progress(self.metadata_file, training_id, progress, metrics)

    def prepare_training_config(self, model_type='yolov5s', epochs=100, batch_size=16, img_size=640, hyp_config=None,
                                freeze_layers=10, evolve_hyp=False, pretrained=True, lr=0.01, patience=30):
        """
        Generate training configuration with enhanced options for small datasets.

        Args:
            model_type: Base model type ('yolov5s', 'yolov5m', etc.)
            epochs: Number of training epochs (default increased to 100 for small datasets)
            batch_size: Batch size (reduced for small datasets)
            img_size: Input image size
            hyp_config: Hyperparameter configuration file path (optional)
            freeze_layers: Number of backbone layers to freeze (helps with small datasets)
            evolve_hyp: Whether to use hyperparameter evolution (helps find optimal settings)
            pretrained: Whether to use pretrained weights
            lr: Learning rate (lowered for small datasets)
            patience: Early stopping patience

        Returns:
            dict: Training configuration
        """
        # Create training configuration
        train_config = {
            "training_id": f"train_{int(time.time())}",
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "dataset_dir": str(self.dataset_dir),
            "output_dir": str(self.output_dir),
            "dataset_yaml": str(self.dataset_dir / "dataset.yaml"),
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "hyp_config": hyp_config,
            "freeze_layers": freeze_layers,
            "evolve_hyp": evolve_hyp,
            "pretrained": pretrained,
            "learning_rate": lr,
            "patience": patience
        }

        # Ensure dataset YAML exists
        if not Path(train_config["dataset_yaml"]).exists():
            self.logger.warning(f"Dataset YAML not found at {train_config['dataset_yaml']}")
            # Try to create it if dataset_manager is available
            try:
                from dataset_manager import DatasetManager
                dataset_manager = DatasetManager(self.dataset_dir)
                dataset_manager.export_dataset_yaml()
                self.logger.info(f"Created dataset YAML at {train_config['dataset_yaml']}")
            except Exception as e:
                self.logger.error(f"Error creating dataset YAML: {e}")
                return None

        # Set model-specific paths
        model_dir = self.output_dir / train_config["training_id"]
        model_dir.mkdir(parents=True, exist_ok=True)

        train_config["model_dir"] = str(model_dir)
        train_config["weights_dir"] = str(model_dir / "weights")
        train_config["best_model_path"] = str(model_dir / "weights" / "best.pt")
        train_config["last_model_path"] = str(model_dir / "weights" / "last.pt")

        # Add to metadata
        self.metadata["training_runs"].append({
            "training_id": train_config["training_id"],
            "model_type": model_type,
            "timestamp": train_config["timestamp"],
            "status": "configured",
            "config": train_config
        })
        self._save_metadata()

        return train_config

    def train(self, training_config=None, model_type='yolov5s', epochs=100, batch_size=8, img_size=640,
              hyp_config=None, use_local=True, freeze_layers=10, evolve_hyp=False, pretrained=True,
              lr=0.01, patience=30):
        """
        Execute training process with enhanced options for small datasets.

        Args:
            training_config: Pre-generated training configuration (optional)
            model_type: Base model type ('yolov5s', 'yolov5m', etc.)
            epochs: Number of training epochs (default increased to 100 for small datasets)
            batch_size: Batch size (reduced for small datasets)
            img_size: Input image size
            hyp_config: Hyperparameter configuration file path (optional)
            use_local: Whether to use local training (True) or remote service (False)
            freeze_layers: Number of backbone layers to freeze (helps with small datasets)
            evolve_hyp: Whether to use hyperparameter evolution (helps find optimal settings)
            pretrained: Whether to use pretrained weights
            lr: Learning rate (lowered for small datasets)
            patience: Early stopping patience

        Returns:
            str: Training ID or None if training failed/is running
        """
        # Check if already training
        if self.is_training:
            self.logger.warning("Training is already in progress")
            return None

        # Prepare training configuration if not provided
        if training_config is None:
            training_config = self.prepare_training_config(
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                img_size=img_size,
                hyp_config=hyp_config,
                freeze_layers=freeze_layers,
                evolve_hyp=evolve_hyp,
                pretrained=pretrained,
                lr=lr,
                patience=patience
            )

        if training_config is None:
            self.logger.error("Failed to prepare training configuration")
            return None

        # Store current training ID
        self.current_training_id = training_config["training_id"]

        # Reset stop event
        self.stop_event.clear()

        # Start training in background thread
        if use_local:
            self.is_training = True
            self.training_thread = threading.Thread(
                target=self._train_local_thread,
                args=(training_config,),
                daemon=True
            )
            self.training_thread.start()
            self.logger.info(f"Started local training thread: {self.current_training_id}")
            return self.current_training_id
        else:
            self.is_training = True
            self.training_thread = threading.Thread(
                target=self._train_remote_thread,
                args=(training_config,),
                daemon=True
            )
            self.training_thread.start()
            self.logger.info(f"Started remote training thread: {self.current_training_id}")
            return self.current_training_id

    def _train_local_thread(self, training_config):
        """
        Training thread for local training with enhanced options for small datasets.

        Args:
            training_config: Training configuration
        """
        try:
            self.logger.info(f"Starting local training: {training_config['training_id']}")

            # Auto-fix validation split if needed
            self._ensure_validation_split()

            # Update training status
            self._update_training_status(training_config["training_id"], "preparing")

            # Clone YOLOv5 repository if needed
            yolov5_dir = self._ensure_yolov5_repo()
            if yolov5_dir is None:
                self.logger.error("Failed to clone or update YOLOv5 repository")
                self._update_training_status(training_config["training_id"], "failed",
                                             {"error": "Failed to clone YOLOv5 repository"})
                self.is_training = False
                return

            # Create weights directory
            weights_dir = Path(training_config["weights_dir"])
            weights_dir.mkdir(parents=True, exist_ok=True)

            # Prepare custom hyperparameter file if not provided
            if not training_config.get("hyp_config"):
                # Create a modified hyperparameter file for small datasets
                custom_hyp_path = weights_dir / "custom_hyp.yaml"
                self._create_small_dataset_hyperparams(custom_hyp_path, lr=training_config.get('learning_rate', 0.01))
                training_config["hyp_config"] = str(custom_hyp_path)

            # Disable Comet ML integration
            os.environ['COMET_MODE'] = 'disabled'

            # Create data augmentation config for small datasets if not already augmented
            augmentation_config = self._create_augmentation_config(weights_dir)

            # Prepare command arguments with correct format and enhanced options
            cmd = [
                sys.executable,
                str(yolov5_dir / "train.py"),
                "--img", str(training_config["img_size"]),
                "--batch-size", str(training_config["batch_size"]),
                "--epochs", str(training_config["epochs"]),
                "--data", training_config["dataset_yaml"],
                "--project", str(self.output_dir),
                "--name", training_config["training_id"],
                "--exist-ok",
                "--save-period", "1",
                "--patience", str(training_config.get("patience", 30)),  # Early stopping patience
                "--freeze", str(training_config.get("freeze_layers", 10))  # Freeze layers
            ]

            # Add weights with correct handling for PyTorch 2.6
            if training_config.get("pretrained", True):
                cmd.extend(["--weights", f"{training_config['model_type']}.pt"])
            else:
                cmd.extend(["--weights", ""])

            # Add hyperparameter file
            if training_config.get("hyp_config"):
                cmd.extend(["--hyp", training_config["hyp_config"]])

            # Add augmentation configuration
            if augmentation_config:
                cmd.extend(["--augment", augmentation_config])

            # Add cache option for small datasets to speed up training
            cmd.extend(["--cache"])

            # Add evolve option if specified
            if training_config.get("evolve_hyp", False):
                cmd.extend(["--evolve"])

            # Log command
            self.logger.info(f"Training command: {' '.join(map(str, cmd))}")

            # Update training status
            self._update_training_status(training_config["training_id"], "training")

            # Create PyTorch 2.6 compatibility wrapper script
            #compatibility_script = self._create_pytorch26_compatibility_script(yolov5_dir)
            compatibility_script = None

            # If compatibility script was created, use it instead
            if compatibility_script:
                cmd = [
                    sys.executable,
                    str(compatibility_script),
                    "--img", str(training_config["img_size"]),
                    "--batch-size", str(training_config["batch_size"]),
                    "--epochs", str(training_config["epochs"]),
                    "--data", training_config["dataset_yaml"],
                    "--project", str(self.output_dir),
                    "--name", training_config["training_id"],
                    "--exist-ok",
                    "--save-period", "1",
                    "--patience", str(training_config.get("patience", 30)),
                    "--freeze", str(training_config.get("freeze_layers", 10))
                ]

                if training_config.get("pretrained", True):
                    cmd.extend(["--weights", f"{training_config['model_type']}.pt"])
                else:
                    cmd.extend(["--weights", ""])

                if training_config.get("hyp_config"):
                    cmd.extend(["--hyp", training_config["hyp_config"]])

                if augmentation_config:
                    cmd.extend(["--augment", augmentation_config])

                cmd.extend(["--cache"])

                if training_config.get("evolve_hyp", False):
                    cmd.extend(["--evolve"])

            # Start training process
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=os.environ.copy()  # Pass environment variables
            )

            # Monitor process output
            metrics = {}
            for line in iter(self.current_process.stdout.readline, ''):
                # Check if training was stopped
                if self.stop_event.is_set():
                    self.current_process.terminate()
                    self.logger.info(f"Training stopped: {training_config['training_id']}")
                    self._update_training_status(training_config["training_id"], "stopped")
                    self.is_training = False
                    return

                # Log output
                self.logger.info(line.strip())

                # Look for model save messages in the output
                if "Saving model" in line or "Saved" in line or ".pt" in line:
                    self.logger.info(f"MODEL SAVE DETECTED: {line.strip()}")

                # Parse metrics from output
                if "Epoch:" in line:
                    try:
                        # Extract key metrics
                        parts = line.strip().split()
                        epoch_idx = parts.index("Epoch:")
                        epoch = int(parts[epoch_idx + 1].split("/")[0])

                        # Find metrics
                        for i, part in enumerate(parts):
                            if part == "mAP@0.5:":
                                metrics["mAP_0.5"] = float(parts[i + 1])
                            elif part == "mAP@0.5:0.95:":
                                metrics["mAP_0.5_0.95"] = float(parts[i + 1])

                        # Update progress
                        progress = epoch / training_config["epochs"]
                        self._update_training_progress(training_config["training_id"], progress, metrics)
                    except Exception as e:
                        self.logger.warning(f"Error parsing metrics: {e}")

            # Wait for process to complete
            self.current_process.wait()

            # Check process return code
            if self.current_process.returncode == 0:
                self.logger.info(f"Training completed successfully: {training_config['training_id']}")

                # Register the trained model with enhanced file finding
                model_info = self._register_trained_model(training_config)

                # Update training status
                self._update_training_status(training_config["training_id"], "completed", {
                    "model_id": model_info["model_id"] if model_info else None,
                    "final_metrics": metrics
                })
            else:
                self.logger.error(
                    f"Training failed with code {self.current_process.returncode}: {training_config['training_id']}")
                self._update_training_status(training_config["training_id"], "failed", {
                    "return_code": self.current_process.returncode
                })

        except Exception as e:
            self.logger.error(f"Error during local training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._update_training_status(training_config["training_id"], "failed", {"error": str(e)})

        finally:
            self.is_training = False
            self.current_process = None
            self.current_training_id = None

    def _create_small_dataset_hyperparams(self, output_path, lr=0.01):
        """
        Create optimized hyperparameters for small datasets.

        Args:
            output_path: Path to save hyperparameter file
            lr: Base learning rate
        """
        # Hyperparameters optimized for small datasets
        small_dataset_hyp = {
            "lr0": lr,  # initial learning rate
            "lrf": 0.01,  # final learning rate = lr0 * lrf
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 5.0,  # longer warmup helps small datasets
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 0.05,
            "cls": 0.5,
            "cls_pw": 1.0,
            "obj": 1.0,
            "obj_pw": 1.0,
            "iou_t": 0.2,
            "anchor_t": 4.0,
            "fl_gamma": 0.0,
            "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
            "degrees": 10.0,  # increased rotation for more augmentation
            "translate": 0.2,  # increased translation
            "scale": 0.5,  # image scale (+/- gain)
            "shear": 2.0,  # added shear for small datasets
            "perspective": 0.0,
            "flipud": 0.1,  # added vertical flip
            "fliplr": 0.5,  # image flip left-right (probability)
            "mosaic": 1.0,  # image mosaic (probability)
            "mixup": 0.25,  # added mixup for small datasets
            "copy_paste": 0.1  # added copy-paste for small datasets
        }

        # Save to YAML file
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(small_dataset_hyp, f)

        return str(output_path)

    def _create_augmentation_config(self, output_dir):
        """
        Create augmentation configuration for small datasets.

        Args:
            output_dir: Directory to save augmentation configuration

        Returns:
            str: Path to augmentation configuration file or None
        """
        # For now, we'll rely on the hyperparameters for augmentation
        # This can be expanded later for more complex augmentation pipelines
        return None

    def _create_pytorch26_compatibility_script(self, yolov5_dir):
        """
        Create a PyTorch 2.6 compatibility wrapper script.

        Args:
            yolov5_dir: Path to YOLOv5 repository

        Returns:
            Path: Path to wrapper script or None if creation failed
        """
        try:
            # Create a wrapper script that adds the necessary compatibility code
            wrapper_path = yolov5_dir / "train_compat.py"

            with open(yolov5_dir / "train.py", 'r') as source:
                content = source.read()

            # Add imports for safe globals
            modified_content = "import torch.serialization\n"
            modified_content += "from models.yolo import DetectionModel\n"
            modified_content += "torch.serialization.add_safe_globals([DetectionModel])\n\n"
            modified_content += content

            # Patch the deprecated torch.cuda.amp.GradScaler warning
            modified_content = modified_content.replace(
                "scaler = torch.cuda.amp.GradScaler(enabled=amp)",
                "try:\n    scaler = torch.amp.GradScaler('cuda', enabled=amp)\nexcept (AttributeError, TypeError):\n    scaler = torch.cuda.amp.GradScaler(enabled=amp)"
            )

            # Patch the deprecated torch.cuda.amp.autocast warning
            modified_content = modified_content.replace(
                "with torch.cuda.amp.autocast(amp):",
                "try:\n        with torch.amp.autocast('cuda', enabled=amp):\n    except (AttributeError, TypeError):\n        with torch.cuda.amp.autocast(amp):"
            )

            with open(wrapper_path, 'w') as target:
                target.write(modified_content)

            return wrapper_path

        except Exception as e:
            self.logger.error(f"Failed to create PyTorch 2.6 compatibility script: {e}")
            return None

    def _train_remote_thread(self, training_config):
        """
        Training thread for remote training.

        Args:
            training_config: Training configuration
        """
        try:
            self.logger.info(f"Starting remote training: {training_config['training_id']}")

            # Update training status
            self._update_training_status(training_config["training_id"], "preparing")

            # Get API endpoint from config
            api_endpoint = self.config.get('training', {}).get('remote_api_endpoint')
            api_key = self.config.get('training', {}).get('remote_api_key')

            if not api_endpoint:
                self.logger.error("Remote training API endpoint not configured")
                self._update_training_status(training_config["training_id"], "failed",
                                             {"error": "Remote training API endpoint not configured"})
                self.is_training = False
                return

            # Package dataset
            import tempfile
            from zipfile import ZipFile
            import requests

            # Create temporary ZIP file
            temp_zip = tempfile.mktemp(suffix='.zip')

            # Create ZIP archive
            with ZipFile(temp_zip, 'w') as zipf:
                dataset_dir = Path(training_config["dataset_dir"])

                # Add all files and directories
                for root, dirs, files in os.walk(dataset_dir):
                    root_path = Path(root)
                    rel_path = root_path.relative_to(dataset_dir)

                    for file in files:
                        file_path = root_path / file
                        arc_path = rel_path / file
                        zipf.write(file_path, arcname=arc_path)

            # Prepare upload request
            files = {
                'dataset': ('dataset.zip', open(temp_zip, 'rb'), 'application/zip')
            }

            data = {
                'model_type': training_config["model_type"],
                'epochs': training_config["epochs"],
                'batch_size': training_config["batch_size"],
                'img_size': training_config["img_size"],
                'training_id': training_config["training_id"]
            }

            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

            # Update training status
            self._update_training_status(training_config["training_id"], "uploading")

            # Upload dataset and start training
            try:
                self.logger.info(f"Uploading dataset to {api_endpoint}/upload")
                response = requests.post(
                    f"{api_endpoint}/upload",
                    files=files,
                    data=data,
                    headers=headers
                )
                response.raise_for_status()

                # Parse response
                result = response.json()
                remote_job_id = result.get('job_id')

                if not remote_job_id:
                    self.logger.error("Failed to get remote job ID")
                    self._update_training_status(training_config["training_id"], "failed",
                                                 {"error": "Invalid response from remote API"})
                    self.is_training = False
                    return

                # Update training status
                self._update_training_status(training_config["training_id"], "training", {
                    "remote_job_id": remote_job_id
                })

                # Poll for job status
                completed = False
                metrics = {}

                while not completed and not self.stop_event.is_set():
                    # Sleep before polling
                    time.sleep(30)  # Poll every 30 seconds

                    # Check job status
                    status_response = requests.get(
                        f"{api_endpoint}/status/{remote_job_id}",
                        headers=headers
                    )
                    status_response.raise_for_status()

                    # Parse status
                    status = status_response.json()
                    job_status = status.get('status', 'unknown')
                    job_progress = status.get('progress', 0.0)
                    job_metrics = status.get('metrics', {})

                    # Update metrics
                    if job_metrics:
                        metrics.update(job_metrics)

                    # Update progress
                    self._update_training_progress(training_config["training_id"], job_progress, metrics)

                    # Check if completed
                    if job_status == 'completed':
                        completed = True
                        model_url = status.get('model_url')

                        if model_url:
                            # Download trained model
                            self.logger.info(f"Downloading trained model from {model_url}")

                            # Create weights directory if it doesn't exist
                            weights_dir = Path(training_config["weights_dir"])
                            weights_dir.mkdir(parents=True, exist_ok=True)

                            # Download model file
                            model_path = weights_dir / "best.pt"

                            model_response = requests.get(model_url, stream=True)
                            model_response.raise_for_status()

                            with open(model_path, 'wb') as f:
                                for chunk in model_response.iter_content(chunk_size=8192):
                                    f.write(chunk)

                            # Register the trained model
                            model_info = self._register_trained_model(training_config, str(model_path))

                            # Update training status
                            self._update_training_status(training_config["training_id"], "completed", {
                                "model_id": model_info["model_id"] if model_info else None,
                                "final_metrics": metrics
                            })
                        else:
                            self.logger.error("Model URL not found in response")
                            self._update_training_status(training_config["training_id"], "failed",
                                                         {"error": "Model URL not found"})

                    elif job_status == 'failed':
                        self.logger.error(f"Remote training failed: {status.get('error', 'Unknown error')}")
                        self._update_training_status(training_config["training_id"], "failed", {
                            "error": status.get('error', 'Unknown error')
                        })
                        completed = True

                    elif job_status == 'stopped':
                        self.logger.info("Remote training was stopped")
                        self._update_training_status(training_config["training_id"], "stopped")
                        completed = True

                # Check if stopped by user
                if self.stop_event.is_set():
                    # Send stop request to remote API
                    requests.post(
                        f"{api_endpoint}/stop/{remote_job_id}",
                        headers=headers
                    )
                    self._update_training_status(training_config["training_id"], "stopped")

            except requests.RequestException as e:
                self.logger.error(f"API request error: {e}")
                self._update_training_status(training_config["training_id"], "failed", {"error": str(e)})

            finally:
                # Clean up dataset package
                if os.path.exists(temp_zip):
                    os.unlink(temp_zip)

        except Exception as e:
            self.logger.error(f"Error during remote training: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self._update_training_status(training_config["training_id"], "failed", {"error": str(e)})

        finally:
            self.is_training = False
            self.current_training_id = None

    def _ensure_validation_split(self):
        """
        Ensure validation split (images/val and labels/val) exists before training.
        If missing, create it and copy a few samples from train.
        Improved to handle more edge cases and ensure proper YOLO structure.
        """
        # Full paths for better logging and debugging
        images_train = self.dataset_dir / 'images' / 'train'
        labels_train = self.dataset_dir / 'labels' / 'train'
        images_val = self.dataset_dir / 'images' / 'val'
        labels_val = self.dataset_dir / 'labels' / 'val'

        # Ensure parent directories exist
        (self.dataset_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / 'labels').mkdir(parents=True, exist_ok=True)

        # Create training directories if they don't exist (though they should)
        images_train.mkdir(parents=True, exist_ok=True)
        labels_train.mkdir(parents=True, exist_ok=True)

        # Create validation directories
        images_val.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

        # Check if validation already has images AND labels
        val_images = list(images_val.glob('*.jpg')) + list(images_val.glob('*.png'))
        val_labels = list(labels_val.glob('*.txt'))

        self.logger.info(f"Found {len(val_images)} validation images and {len(val_labels)} validation labels")

        # Only consider validation set valid if it has both images and labels
        if len(val_images) == 0 or len(val_labels) == 0:
            self.logger.info("[AutoFix] Validation set missing or incomplete, creating from training data")

            # Get training images and labels
            train_images = list(images_train.glob('*.jpg')) + list(images_train.glob('*.png'))
            train_labels = list(labels_train.glob('*.txt'))

            if len(train_images) == 0 or len(train_labels) == 0:
                self.logger.error("[AutoFix] No training images or labels found to create validation set")
                return False

            # Clear existing validation data for consistency
            for img in val_images:
                img.unlink(missing_ok=True)
            for lbl in val_labels:
                lbl.unlink(missing_ok=True)

            # Copy samples to validation (minimum 5, maximum 20, target 10% of dataset)
            sample_count = min(max(5, int(len(train_images) * 0.1)), 20)
            self.logger.info(f"[AutoFix] Creating validation set with {sample_count} samples")

            # Track successful copies to report accurate numbers
            successful_copies = 0

            for img_path in train_images[:sample_count]:
                # Copy image
                target_path = images_val / img_path.name
                try:
                    shutil.copy(img_path, target_path)

                    # Find and copy corresponding label file
                    base_name = img_path.stem
                    label_path = labels_train / f"{base_name}.txt"

                    if label_path.exists():
                        target_label = labels_val / f"{base_name}.txt"
                        shutil.copy(label_path, target_label)
                        successful_copies += 1
                    else:
                        self.logger.warning(f"[AutoFix] No label file found for {img_path.name}")
                except Exception as e:
                    self.logger.error(f"[AutoFix] Error copying file {img_path}: {e}")

            # Verify the split was successful
            new_val_images = list(images_val.glob('*.jpg')) + list(images_val.glob('*.png'))
            new_val_labels = list(labels_val.glob('*.txt'))

            self.logger.info(
                f"[AutoFix] Created validation set with {len(new_val_images)} images and {len(new_val_labels)} labels")

            # Check validity of created validation set
            if len(new_val_images) == 0 or len(new_val_labels) == 0:
                self.logger.error("[AutoFix] Failed to create valid validation set")
                return False

            return True
        else:
            self.logger.info(
                f"[AutoFix] Validation set already exists with {len(val_images)} images and {len(val_labels)} labels")
            return True

    def _ensure_yolov5_repo(self):
        """
        Ensure YOLOv5 repository is available.

        Returns:
            Path: Path to YOLOv5 repository or None if failed
        """
        try:
            # Set up YOLOv5 directory in output directory
            yolov5_dir = self.output_dir / "yolov5"

            # Check if directory exists
            if yolov5_dir.exists():
                # Update repository
                self.logger.info(f"Updating YOLOv5 repository at {yolov5_dir}")
                try:
                    subprocess.run(
                        ["git", "pull"],
                        cwd=yolov5_dir,
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to update repository: {e}. Will use existing version.")
            else:
                # Clone repository
                self.logger.info(f"Cloning YOLOv5 repository to {yolov5_dir}")
                subprocess.run(
                    ["git", "clone", "https://github.com/ultralytics/yolov5.git", str(yolov5_dir)],
                    check=True,
                    capture_output=True
                )

            # Install dependencies
            self.logger.info("Installing YOLOv5 dependencies")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(yolov5_dir / "requirements.txt")],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Failed to install all dependencies: {e}. Training might still work.")

            return yolov5_dir

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.cmd} (exit code {e.returncode})")
            if hasattr(e, 'stderr'):
                self.logger.error(f"Error output: {e.stderr}")
            return None

        except Exception as e:
            self.logger.error(f"Error ensuring YOLOv5 repository: {e}")
            return None

    def _register_trained_model(self, training_config, model_path=None):
        """
        Register trained model in model registry with PyTorch 2.6 compatibility.

        Args:
            training_config: Training configuration
            model_path: Path to model file (if different from default)

        Returns:
            dict: Model information or None if registration failed
        """
        try:
            # Use default model path if not specified
            if model_path is None:
                # Define expected paths
                expected_best = training_config["best_model_path"]
                expected_last = training_config["last_model_path"]

                # Check expected paths first
                if os.path.exists(expected_best):
                    model_path = expected_best
                    self.logger.info(f"Using best model: {model_path}")
                elif os.path.exists(expected_last):
                    model_path = expected_last
                    self.logger.info(f"Using last model: {model_path}")
                else:
                    # Try to find where YOLOv5 actually saved the files
                    train_dir = Path(self.output_dir) / training_config["training_id"]
                    self.logger.info(f"Searching for model files in training directory: {train_dir}")

                    # Standard YOLOv5 paths
                    alternative_paths = [
                        train_dir / "weights" / "best.pt",
                        train_dir / "weights" / "last.pt",
                        train_dir / "best.pt",
                        train_dir / "last.pt",
                        Path(self.output_dir) / "yolov5" / "runs" / "train" / training_config[
                            "training_id"] / "weights" / "best.pt",
                        Path(self.output_dir) / "yolov5" / "runs" / "train" / training_config[
                            "training_id"] / "weights" / "last.pt"
                    ]

                    # Check all alternative paths
                    for path in alternative_paths:
                        if os.path.exists(path):
                            model_path = str(path)
                            self.logger.info(f"Found model at alternative location: {model_path}")
                            break

                    # If still not found, search recursively
                    if model_path is None:
                        self.logger.info("Searching recursively for model files...")
                        pt_files = []

                        # Search in training directory
                        for root, dirs, files in os.walk(train_dir):
                            for file in files:
                                if file.endswith('.pt'):
                                    pt_path = os.path.join(root, file)
                                    pt_files.append((pt_path, os.path.getmtime(pt_path)))

                        # Sort by modification time (newest first)
                        pt_files.sort(key=lambda x: x[1], reverse=True)

                        if pt_files:
                            model_path = pt_files[0][0]
                            self.logger.info(f"Using most recently modified model: {model_path}")
                        else:
                            self.logger.error(f"No model files found in {train_dir}")
                            return None

            # Check if model file exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return None

            # Create model directory in registry
            model_dir = self.output_dir / "registry" / f"model_{int(time.time())}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Copy model file
            model_file = model_dir / "model.pt"
            shutil.copy(model_path, model_file)
            self.logger.info(f"Copied model from {model_path} to {model_file}")

            # Export to ONNX if exporter is available
            onnx_file = None
            if self.exporter is not None:
                try:
                    onnx_file = self.exporter.export_model(str(model_file), 'onnx')
                except Exception as e:
                    self.logger.warning(f"Failed to convert model to ONNX: {e}")

            # Register model in metadata
            model_info = {
                "model_id": f"model_{int(time.time())}_{training_config['model_type']}",
                "training_id": training_config["training_id"],
                "model_type": training_config["model_type"],
                "model_path": str(model_file),
                "onnx_path": onnx_file,
                "created": time.time(),
                "dataset_stats": self._get_dataset_stats(training_config["dataset_dir"]),
                "training_config": training_config,
                "source_model_path": model_path
            }

            # Add to metadata
            self.metadata["models"].append(model_info)
            self._save_metadata()

            # Load model and get information with PyTorch 2.6 compatibility
            try:
                # Add safe globals for PyTorch 2.6+
                try:
                    from torch.serialization import add_safe_globals
                    import sys
                    sys.path.append(str(self.output_dir / "yolov5"))  # Make sure YOLOv5 modules are discoverable
                    from models.yolo import DetectionModel  # Pre-import DetectionModel for torch.load compatibility

                    add_safe_globals([DetectionModel])
                    model = torch.load(model_path, map_location=torch.device('cpu'))
                except (ImportError, AttributeError):
                    # Fallback for older PyTorch versions or if module not found
                    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

                if isinstance(model, dict) and "model" in model:
                    model_state = model["model"].state_dict() if hasattr(model["model"], "state_dict") else None
                    model_info["model_size"] = sum(param.numel() for param in model["model"].parameters())
                    model_info["layers"] = len(list(model["model"].parameters()))

                    if hasattr(model["model"], "names") and model["model"].names:
                        model_info["classes"] = model["model"].names

                if isinstance(model, dict) and "epoch" in model:
                    model_info["trained_epochs"] = model["epoch"]

                if isinstance(model, dict) and "best_fitness" in model:
                    model_info["best_fitness"] = float(model["best_fitness"])

            except Exception as e:
                self.logger.warning(f"Error extracting model information: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())

            # Update metadata with model details
            for i, m in enumerate(self.metadata["models"]):
                if m["model_id"] == model_info["model_id"]:
                    self.metadata["models"][i] = model_info
                    break

            self._save_metadata()

            self.logger.info(f"Registered trained model: {model_info['model_id']}")
            return model_info

        except Exception as e:
            self.logger.error(f"Error registering trained model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _get_dataset_stats(self, dataset_dir):
        """
        Get dataset statistics.

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            dict: Dataset statistics
        """
        try:
            # Try to use dataset manager if available
            try:
                from dataset_manager import DatasetManager
                dataset_manager = DatasetManager(dataset_dir)
                return dataset_manager.get_statistics()
            except ImportError:
                pass

            # Simple statistics if dataset manager not available
            dataset_dir = Path(dataset_dir)

            stats = {
                "dataset_dir": str(dataset_dir)
            }

            # Count images in each split
            for split in ['train', 'val', 'test']:
                split_dir = dataset_dir / 'images' / split
                if split_dir.exists():
                    stats[f"{split}_images"] = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
                else:
                    stats[f"{split}_images"] = 0

            # Count annotations in each split
            for split in ['train', 'val', 'test']:
                split_dir = dataset_dir / 'labels' / split
                if split_dir.exists():
                    # Count lines in all text files
                    count = 0
                    for label_file in split_dir.glob('*.txt'):
                        try:
                            with open(label_file, 'r') as f:
                                count += sum(1 for line in f)
                        except:
                            pass
                    stats[f"{split}_annotations"] = count
                else:
                    stats[f"{split}_annotations"] = 0

            # Get class list if available
            classes_file = dataset_dir / 'classes.txt'
            if classes_file.exists():
                try:
                    with open(classes_file, 'r') as f:
                        stats["classes"] = [line.strip() for line in f if line.strip()]
                    stats["class_count"] = len(stats["classes"])
                except:
                    pass

            # Calculate class distribution to identify imbalance
            class_distribution = {}
            for split in ['train', 'val', 'test']:
                labels_dir = dataset_dir / 'labels' / split
                if not labels_dir.exists():
                    continue

                for label_file in labels_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if class_id not in class_distribution:
                                        class_distribution[class_id] = 0
                                    class_distribution[class_id] += 1
                    except:
                        pass

            if class_distribution:
                stats["class_distribution"] = class_distribution

            return stats

        except Exception as e:
            self.logger.error(f"Error getting dataset stats: {e}")
            return {"error": str(e)}

    def evaluate(self, model_id=None, model_path=None, dataset_yaml=None):
        """
        Evaluate model on validation/test dataset.

        Args:
            model_id: Model ID in registry
            model_path: Direct path to model file (if model_id not provided)
            dataset_yaml: Path to dataset YAML file (defaults to main dataset)

        Returns:
            dict: Evaluation metrics
        """
        if self.evaluator is None:
            self.logger.error("Evaluator component not initialized")
            return {"error": "Evaluator component not initialized"}

        return self.evaluator.evaluate(model_id, model_path, dataset_yaml)

    def export_model(self, model_id, format='onnx', output_path=None, img_size=640):
        """
        Export trained model to deployment format.

        Args:
            model_id: Model ID in registry
            format: Export format ('onnx', 'tflite', 'trt', etc.)
            output_path: Path to save exported model
            img_size: Input image size for the model

        Returns:
            str: Path to exported model or None if export failed
        """
        if self.exporter is None:
            self.logger.error("Exporter component not initialized")
            return None

        return self.exporter.export_model(model_id, format, output_path, img_size)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Model Trainer CLI")
    parser.add_argument("--action", type=str, required=True, help="Action to perform: train, evaluate, export")
    parser.add_argument("--model-type", type=str, default="yolov5s", help="Model type (e.g., yolov5s, yolov5m)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--freeze", type=int, default=10, help="Number of layers to freeze")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="models", help="Path to output directory")
    parser.add_argument("--hyp-config", type=str, default=None, help="Path to hyperparameter YAML (optional)")
    parser.add_argument("--use-local", action="store_true",
                        help="Force local training instead of remote")  # <<< only once here

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = {}  # Additional config can be loaded if needed

    trainer = ModelTrainer(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        config=config
    )

    if args.action == "train":
        training_id = trainer.train(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            hyp_config=args.hyp_config,
            use_local=args.use_local,
            freeze_layers=args.freeze,
            lr=args.lr,
            patience=args.patience
        )

        if training_id:
            # Wait for training thread to finish
            if trainer.training_thread is not None:
                trainer.training_thread.join()

            print(f"\n✅ Training completed for {training_id}")

            # 📈 Generate training graphs after training
            trainer._generate_training_graphs(Path(trainer.output_dir) / training_id)

            # Then proceed to export and evaluation
            model_id = None
            for model_info in trainer.metadata.get("models", []):
                if model_info.get("training_id") == training_id:
                    model_id = model_info.get("model_id")
                    break

            if model_id:
                print(f"📦 Exporting model {model_id} to ONNX format...")

                exported_path = trainer.export_model(model_id, format="onnx")
                if exported_path:
                    print(f"🎯 Model successfully exported to: {exported_path}")

                    # 🧪 Auto-Evaluate after export
                    print(f"🧪 Evaluating model {model_id}...")
                    eval_results = trainer.evaluate(model_id=model_id)

                    if eval_results:
                        eval_report_path = Path(trainer.output_dir) / "registry" / model_id / "evaluation.json"
                        eval_report_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(eval_report_path, "w") as f:
                            json.dump(eval_results, f, indent=2)
                        print(f"📄 Evaluation report saved to {eval_report_path}")

                        # 🗂 Create full summary report
                        summary = {
                            "training_id": training_id,
                            "model_id": model_id,
                            "model_type": args.model_type,
                            "epochs": args.epochs,
                            "batch_size": args.batch_size,
                            "image_size": args.img_size,
                            "learning_rate": args.lr,
                            "freeze_layers": args.freeze,
                            "early_stopping_patience": args.patience,
                            "exported_model_path": str(exported_path),
                            "evaluation": eval_results,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                        }

                        summary_path = Path(trainer.output_dir) / "registry" / model_id / "summary.json"
                        with open(summary_path, "w") as f:
                            json.dump(summary, f, indent=2)

                        print(f"🗂 Summary report saved to {summary_path}")

                    else:
                        print(f"⚠️ Evaluation failed for {model_id}")

                else:
                    print(f"⚠️ Model export failed for {model_id}")
            else:
                print("⚠️ Could not find model ID to export!")




