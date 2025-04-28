#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fix Dataset Classes for YOLOv5 Training.

This script analyzes your dataset labels and updates the dataset.yaml file
to include all class IDs found in your labels.
"""

import os
import logging
import yaml
from pathlib import Path
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fix_classes')


def analyze_and_fix_dataset_classes(dataset_dir="dataset"):
    """
    Analyze dataset labels to find all class IDs and update dataset.yaml accordingly.

    Args:
        dataset_dir: Path to dataset directory (default: "dataset")

    Returns:
        bool: Success status
    """
    try:
        dataset_path = Path(dataset_dir)
        logger.info(f"Analyzing dataset classes in {dataset_path}")

        # Find all label files
        labels_path = dataset_path / "labels"
        if not labels_path.exists():
            logger.error(f"Labels directory not found: {labels_path}")
            return False

        # Collect all class IDs from labels
        class_ids = set()
        class_counts = Counter()
        label_files = []

        # Search in train, val, and test directories
        for split in ['train', 'val', 'test']:
            split_dir = labels_path / split
            if split_dir.exists():
                split_files = list(split_dir.glob('*.txt'))
                label_files.extend(split_files)
                logger.info(f"Found {len(split_files)} label files in {split} split")

        if not label_files:
            logger.error("No label files found in dataset")
            return False

        # Extract class IDs from label files
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 5:  # YOLO format: class_id x y w h
                            try:
                                class_id = int(parts[0])
                                class_ids.add(class_id)
                                class_counts[class_id] += 1
                            except ValueError:
                                logger.warning(f"Invalid class ID in {label_file}")
            except Exception as e:
                logger.warning(f"Error reading {label_file}: {e}")

        if not class_ids:
            logger.error("No valid class IDs found in labels")
            return False

        # Get max class ID to determine number of classes needed
        max_class_id = max(class_ids)
        num_classes = max_class_id + 1

        logger.info(f"Found class IDs: {sorted(class_ids)}")
        logger.info(f"Class distribution: {class_counts}")
        logger.info(f"Number of classes needed: {num_classes}")

        # Try to get class names from classes.txt file
        class_names = []
        classes_file = dataset_path / "classes.txt"

        if classes_file.exists():
            try:
                with open(classes_file, 'r') as f:
                    class_names = [line.strip() for line in f if line.strip()]
                logger.info(f"Found {len(class_names)} class names in classes.txt")

                # Ensure we have enough class names
                if len(class_names) < num_classes:
                    logger.warning(f"classes.txt has {len(class_names)} names but we need {num_classes}")
                    # Extend class names list
                    for i in range(len(class_names), num_classes):
                        class_names.append(f"class_{i}")

                    # Update classes.txt file
                    with open(classes_file, 'w') as f:
                        for name in class_names:
                            f.write(f"{name}\n")
                    logger.info(f"Updated classes.txt with {len(class_names)} classes")
            except Exception as e:
                logger.error(f"Error reading/writing classes.txt: {e}")
                class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            # Create classes.txt file if it doesn't exist
            class_names = [f"class_{i}" for i in range(num_classes)]
            try:
                with open(classes_file, 'w') as f:
                    for name in class_names:
                        f.write(f"{name}\n")
                logger.info(f"Created classes.txt with {len(class_names)} classes")
            except Exception as e:
                logger.error(f"Error creating classes.txt: {e}")

        # Update dataset.yaml file
        yaml_file = dataset_path / "dataset.yaml"
        yaml_content = {}

        if yaml_file.exists():
            try:
                with open(yaml_file, 'r') as f:
                    yaml_content = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error reading dataset.yaml: {e}")

        # Update YAML content
        yaml_content['path'] = str(dataset_path.absolute())
        yaml_content['train'] = 'images/train'
        yaml_content['val'] = 'images/val'
        yaml_content['nc'] = num_classes
        yaml_content['names'] = class_names[:num_classes]  # Ensure correct length

        # Save updated YAML file
        try:
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_content, f, default_flow_style=False)
            logger.info(f"Updated dataset.yaml with {num_classes} classes")
        except Exception as e:
            logger.error(f"Error writing dataset.yaml: {e}")
            return False

        logger.info("✅ Successfully fixed dataset classes")
        return True

    except Exception as e:
        logger.error(f"Error analyzing dataset classes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def check_label_format(dataset_dir="dataset"):
    """
    Check if any label files have incorrect format and fix them.
    Sometimes label files can have incorrect values or extra whitespace.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        int: Number of fixed files
    """
    try:
        dataset_path = Path(dataset_dir)
        labels_path = dataset_path / "labels"
        fixed_count = 0

        if not labels_path.exists():
            logger.error(f"Labels directory not found: {labels_path}")
            return 0

        # Search in train, val, and test directories
        for split in ['train', 'val', 'test']:
            split_dir = labels_path / split
            if not split_dir.exists():
                continue

            label_files = list(split_dir.glob('*.txt'))
            logger.info(f"Checking {len(label_files)} label files in {split} split")

            for label_file in label_files:
                try:
                    fixed_lines = []
                    needs_fixing = False

                    with open(label_file, 'r') as f:
                        for i, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                # Skip empty lines
                                continue

                            parts = line.split()
                            if len(parts) < 5:
                                logger.warning(f"Line {i + 1} in {label_file} has fewer than 5 values: {line}")
                                needs_fixing = True
                                continue

                            try:
                                # Parse and validate values
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])

                                # Validate bounding box coordinates (should be between 0 and 1)
                                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                    logger.warning(f"Invalid coordinates in {label_file}, line {i + 1}: {line}")
                                    # Try to fix coordinates by clamping to [0, 1]
                                    x = max(0, min(x, 1))
                                    y = max(0, min(y, 1))
                                    w = max(0, min(w, 1))
                                    h = max(0, min(h, 1))
                                    needs_fixing = True

                                # Add fixed line
                                fixed_lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                            except ValueError:
                                logger.warning(f"Invalid values in {label_file}, line {i + 1}: {line}")
                                needs_fixing = True

                    if needs_fixing and fixed_lines:
                        with open(label_file, 'w') as f:
                            for line in fixed_lines:
                                f.write(line + '\n')
                        fixed_count += 1
                        logger.info(f"Fixed formatting in {label_file}")

                except Exception as e:
                    logger.warning(f"Error processing {label_file}: {e}")

        logger.info(f"Fixed formatting in {fixed_count} label files")
        return fixed_count

    except Exception as e:
        logger.error(f"Error checking label format: {e}")
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fix Dataset Classes for YOLOv5 Training")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--check-labels", action="store_true", help="Check and fix label formatting issues")

    args = parser.parse_args()

    if args.check_labels:
        fixed_count = check_label_format(args.dataset)
        if fixed_count > 0:
            logger.info(f"Fixed formatting issues in {fixed_count} label files")

    success = analyze_and_fix_dataset_classes(args.dataset)

    if success:
        print("\n✅ Dataset classes successfully fixed. You can now run training again.")
        print("   The dataset.yaml file has been updated with the correct number of classes.")
    else:
        print("\n❌ Failed to fix dataset classes. Please check logs for details.")