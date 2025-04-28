#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration example for ModelTrainer with Dataset Size Handler.
Shows how to use dataset size configurations in the training process.
"""

import sys
import os
import yaml
import logging
from pathlib import Path

# Import the ModelTrainer (assuming it's in the same directory)
# from model_trainer import ModelTrainer

# Import our new DatasetSizeHandler
from dataset_size_handler import DatasetSizeHandler


def train_with_dataset_size_config():
    """Example function showing how to use dataset size configurations for training"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('train_script')

    # Paths
    config_path = "settings.yaml"
    dataset_dir = "dataset"
    output_dir = "models/output"

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    # Initialize the DatasetSizeHandler
    size_handler = DatasetSizeHandler(config_path)

    # Determine dataset size type (small, medium, large)
    dataset_size_type = size_handler.get_dataset_size_type(dataset_dir)
    logger.info(f"Detected dataset size: {dataset_size_type}")

    # Get appropriate training configuration
    training_config = size_handler.get_training_config(dataset_dir)
    logger.info(f"Using training configuration: {training_config}")

    # Initialize the ModelTrainer
    model_trainer = ModelTrainer(dataset_dir, output_dir, config)

    # Prepare training configuration with dataset-specific settings
    train_config = model_trainer.prepare_training_config(
        model_type=training_config.get('model_type', 'yolov5s'),
        epochs=training_config.get('epochs', 100),
        batch_size=training_config.get('batch_size', 8),
        img_size=training_config.get('img_size', 640),
        freeze_layers=training_config.get('freeze_layers', 10),
        evolve_hyp=training_config.get('evolve_hyp', False),
        pretrained=training_config.get('pretrained', True),
        lr=training_config.get('lr', 0.01),
        patience=training_config.get('patience', 30)
    )

    if train_config is None:
        logger.error("Failed to prepare training configuration")
        return

    # Start training with the prepared configuration
    training_id = model_trainer.train(training_config=train_config)

    if training_id:
        logger.info(f"Training started with ID: {training_id}")
        logger.info(f"Training configuration: {train_config}")
    else:
        logger.error("Failed to start training")


def train_explicitly_with_small_dataset_config():
    """Example function showing how to explicitly use small dataset configuration"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('train_script')

    # Paths
    config_path = "settings.yaml"
    dataset_dir = "dataset"
    output_dir = "models/output"

    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return

    # Initialize the ModelTrainer
    model_trainer = ModelTrainer(dataset_dir, output_dir, config)

    # Initialize the DatasetSizeHandler
    size_handler = DatasetSizeHandler(config_path)

    # Explicitly get small dataset configuration
    small_dataset_config = size_handler.get_training_config(size_type='small_dataset')

    # Start training with small dataset configuration
    training_id = model_trainer.train(
        model_type=small_dataset_config.get('model_type', 'yolov5s'),
        epochs=small_dataset_config.get('epochs', 100),
        batch_size=small_dataset_config.get('batch_size', 8),
        img_size=small_dataset_config.get('img_size', 640),
        freeze_layers=small_dataset_config.get('freeze_layers', 10),
        evolve_hyp=small_dataset_config.get('evolve_hyp', True),
        pretrained=small_dataset_config.get('pretrained', True),
        lr=small_dataset_config.get('lr', 0.01),
        patience=small_dataset_config.get('patience', 30)
    )

    if training_id:
        logger.info(f"Training started with ID: {training_id} using small dataset configuration")
    else:
        logger.error("Failed to start training")


# Run if executed directly
if __name__ == "__main__":
    # Choose which example to run
    if len(sys.argv) > 1 and sys.argv[1] == 'small':
        train_explicitly_with_small_dataset_config()
    else:
        train_with_dataset_size_config()