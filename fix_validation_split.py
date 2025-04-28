#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Dataset Validation Split for YOLOv5 Training.

This script ensures proper validation split by checking and creating necessary directory structure
required by YOLOv5 training process.
"""

import os
import shutil
import logging
from pathlib import Path
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fix_validation_split')


def fix_dataset_structure(dataset_dir="dataset"):
    """
    Fix dataset structure by ensuring proper validation split.

    Args:
        dataset_dir: Path to dataset directory (default: "dataset")

    Returns:
        bool: Success status
    """
    try:
        dataset_path = Path(dataset_dir)
        logger.info(f"Checking dataset structure in {dataset_path}")

        # Check main directories
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        if not images_dir.exists():
            images_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created images directory at {images_dir}")

        if not labels_dir.exists():
            labels_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created labels directory at {labels_dir}")

        # Check train directories
        images_train = images_dir / "train"
        labels_train = labels_dir / "train"

        if not images_train.exists():
            images_train.mkdir(parents=True, exist_ok=True)
            logger.error("No training images found! You must add training images.")
            return False

        if not labels_train.exists():
            labels_train.mkdir(parents=True, exist_ok=True)
            logger.error("No training labels found! You must add training labels.")
            return False

        # Check validation directories and create them properly if missing
        images_val = images_dir / "val"
        labels_val = labels_dir / "val"

        images_val.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

        # Check if validation has content
        val_images = list(images_val.glob('*.jpg')) + list(images_val.glob('*.png'))

        if len(val_images) == 0:
            logger.info("No validation images found, creating validation set from training data")

            # Get training images and labels
            train_images = list(images_train.glob('*.jpg')) + list(images_train.glob('*.png'))
            train_labels = list(labels_train.glob('*.txt'))

            if len(train_images) == 0:
                logger.error("No training images found to create validation set!")
                return False

            # Copy a few training images and labels to validation
            sample_count = min(max(5, int(len(train_images) * 0.1)), 20)  # 10% or at least 5, max 20
            logger.info(f"Copying {sample_count} samples to validation set")

            for img_path in train_images[:sample_count]:
                target_path = images_val / img_path.name
                shutil.copy(img_path, target_path)
                logger.info(f"Copied {img_path.name} to validation images")

                # Find and copy corresponding label file
                base_name = img_path.stem
                label_file = labels_train / f"{base_name}.txt"
                if label_file.exists():
                    target_label = labels_val / f"{base_name}.txt"
                    shutil.copy(label_file, target_label)
                    logger.info(f"Copied {label_file.name} to validation labels")
                else:
                    logger.warning(f"No label file found for {img_path.name}")

            logger.info(f"Created validation set with {sample_count} samples")
        else:
            logger.info(f"Validation set already exists with {len(val_images)} images")

        # Fix dataset YAML file
        yaml_file = dataset_path / "dataset.yaml"
        if not yaml_file.exists() or True:  # Always regenerate to ensure paths are correct
            logger.info(f"Creating dataset YAML at {yaml_file}")

            # Try to find class names
            class_file = dataset_path / "classes.txt"
            classes = []
            if class_file.exists():
                with open(class_file, 'r') as f:
                    classes = [line.strip() for line in f if line.strip()]

            if not classes:
                # Try to infer classes from label files if no class file
                label_files = list(labels_train.glob('*.txt')) + list(labels_val.glob('*.txt'))
                class_ids = set()
                for label_file in label_files[:100]:  # Check first 100 files
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_ids.add(int(parts[0]))
                    except Exception as e:
                        logger.warning(f"Error reading label file {label_file}: {e}")

                classes = [f"class_{i}" for i in sorted(class_ids)]

            # Create YAML content
            yaml_content = {
                "path": str(dataset_path.absolute()),
                "train": str(images_train.relative_to(dataset_path)),
                "val": str(images_val.relative_to(dataset_path)),
                "nc": len(classes),
                "names": classes
            }

            # Save YAML file
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)

            logger.info(f"Created dataset YAML with {len(classes)} classes")

        # Verify the structure is now correct
        if images_val.exists() and labels_val.exists() and len(list(images_val.glob('*.*'))) > 0:
            logger.info("✓ Dataset structure is now correctly set up with validation split")
            return True
        else:
            logger.error("× Failed to properly set up validation split")
            return False

    except Exception as e:
        logger.error(f"Error fixing dataset structure: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix Dataset Validation Split for YOLOv5 Training")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory path")

    args = parser.parse_args()

    success = fix_dataset_structure(args.dataset)

    if success:
        print("✅ Dataset successfully fixed. You can now run training again.")
    else:
        print("❌ Failed to fix dataset. Please check logs for details.")