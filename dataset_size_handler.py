#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Size Handler for Smart Object Tracking System.
Module for determining dataset size and applying appropriate training configurations.
"""

import os
import yaml
from pathlib import Path
import logging


class DatasetSizeHandler:
    """Handler for dataset size detection and configuration selection"""

    def __init__(self, config_path="settings.yaml"):
        """
        Initialize the dataset size handler.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger('DatasetSizeHandler')
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}

    def get_dataset_size_type(self, dataset_dir):
        """
        Determine if a dataset is small, medium, or large.

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            str: 'small_dataset', 'large_dataset', or 'medium_dataset'
        """
        dataset_dir = Path(dataset_dir)

        # Count images in train directory
        train_dir = dataset_dir / 'images' / 'train'
        if not train_dir.exists():
            self.logger.warning(f"Train directory not found: {train_dir}")
            return 'small_dataset'  # Default to small dataset config if directory structure not found

        # Count image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_count = sum(1 for _ in train_dir.glob('*') if _.suffix.lower() in image_extensions)

        self.logger.info(f"Found {image_count} images in training set")

        # Define thresholds for dataset sizes
        if image_count < 500:
            return 'small_dataset'
        elif image_count > 5000:
            return 'large_dataset'
        else:
            return 'medium_dataset'  # Default to regular settings

    def get_training_config(self, dataset_dir=None, size_type=None):
        """
        Get appropriate training configuration based on dataset size.

        Args:
            dataset_dir: Path to dataset directory
            size_type: Override automatic size detection with specific type

        Returns:
            dict: Training configuration for the dataset size
        """
        if size_type is None and dataset_dir is not None:
            size_type = self.get_dataset_size_type(dataset_dir)
        elif size_type is None:
            size_type = 'medium_dataset'  # Default if no dataset_dir provided

        self.logger.info(f"Using {size_type} configuration")

        # Start with default training configuration
        training_config = {
            "model_type": self.config.get('training', {}).get('default_model_type', 'yolov5s'),
            "epochs": self.config.get('training', {}).get('default_epochs', 50),
            "batch_size": self.config.get('training', {}).get('default_batch_size', 16),
            "img_size": self.config.get('training', {}).get('default_img_size', 640),
            "patience": self.config.get('training', {}).get('patience', 20),
            "freeze_layers": 0,
            "evolve_hyp": False,
            "pretrained": True,
            "lr": 0.01
        }

        # Override with dataset-specific configurations if available
        dataset_configs = self.config.get('training', {}).get('dataset_configs', {})
        if size_type in dataset_configs:
            specific_config = dataset_configs[size_type]

            # Update configuration with dataset-specific values
            if 'lr' in specific_config:
                training_config['lr'] = specific_config['lr']
            if 'batch_size' in specific_config:
                training_config['batch_size'] = specific_config['batch_size']
            if 'epochs' in specific_config:
                training_config['epochs'] = specific_config['epochs']
            if 'freeze_layers' in specific_config:
                training_config['freeze_layers'] = specific_config['freeze_layers']
            if 'patience' in specific_config:
                training_config['patience'] = specific_config['patience']
            if 'evolve_hyp' in specific_config:
                training_config['evolve_hyp'] = specific_config['evolve_hyp']

            # Include any additional settings
            for key, value in specific_config.items():
                if key not in training_config:
                    training_config[key] = value

        return training_config


# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    handler = DatasetSizeHandler("settings.yaml")

    # Example dataset path
    dataset_path = "dataset"

    # Automatically determine dataset size and get appropriate configuration
    config = handler.get_training_config(dataset_path)
    print(f"Automatically determined configuration: {config}")

    # Or explicitly specify dataset size type
    small_config = handler.get_training_config(size_type='small_dataset')
    print(f"Small dataset configuration: {small_config}")

    large_config = handler.get_training_config(size_type='large_dataset')
    print(f"Large dataset configuration: {large_config}")