#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Label Statistics and Fix Tool for YOLOv5 Training.

This script provides detailed statistics on your label files and can fix common issues.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt


def analyze_labels(dataset_dir="dataset", fix=False, visualize=False):
    """
    Analyze label files and provide detailed statistics.

    Args:
        dataset_dir: Path to dataset directory
        fix: Whether to fix issues automatically
        visualize: Whether to create visualization plots

    Returns:
        dict: Analysis results
    """
    dataset_path = Path(dataset_dir)
    labels_path = dataset_path / "labels"

    results = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "empty_files": 0,
        "class_distribution": {},
        "bbox_sizes": [],
        "aspect_ratios": [],
        "class_ids": set(),
        "issues": [],
        "fixes": [],
    }

    # Check if labels directory exists
    if not labels_path.exists():
        print(f"❌ Labels directory not found: {labels_path}")
        results["issues"].append(f"Labels directory not found: {labels_path}")
        return results

    # Process each split directory
    class_dist_by_split = {}
    file_counts_by_split = {}

    for split in ['train', 'val', 'test']:
        split_dir = labels_path / split
        if not split_dir.exists():
            continue

        label_files = list(split_dir.glob('*.txt'))
        file_counts_by_split[split] = len(label_files)
        results["total_files"] += len(label_files)

        class_dist_by_split[split] = Counter()
        invalid_files = []
        empty_files = []

        # Process each label file
        for label_file in label_files:
            try:
                file_has_valid_labels = False
                file_has_errors = False
                fixed_lines = []

                with open(label_file, 'r') as f:
                    lines = f.readlines()

                if not lines:
                    empty_files.append(label_file)
                    results["empty_files"] += 1
                    continue

                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        print(f"❌ Invalid format in {label_file}, line {i + 1}: {line}")
                        results["issues"].append(f"Invalid format in {label_file}, line {i + 1}: {line}")
                        file_has_errors = True
                        continue

                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])

                        # Check if values are valid
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            print(f"❌ Invalid coordinates in {label_file}, line {i + 1}: {line}")
                            results["issues"].append(f"Invalid coordinates in {label_file}, line {i + 1}")
                            file_has_errors = True

                            if fix:
                                # Fix by clamping values
                                x = max(0, min(x, 1))
                                y = max(0, min(y, 1))
                                w = max(0, min(w, 1))
                                h = max(0, min(h, 1))
                                fixed_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                                fixed_lines.append(fixed_line)
                                results["fixes"].append(f"Fixed coordinates in {label_file}, line {i + 1}")
                            continue

                        # Store valid data for analysis
                        class_dist_by_split[split][class_id] += 1
                        results["class_ids"].add(class_id)
                        results["bbox_sizes"].append(w * h)
                        results["aspect_ratios"].append(w / h if h > 0 else 0)
                        file_has_valid_labels = True

                        if fix:
                            # Normalize format for consistency
                            fixed_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                            fixed_lines.append(fixed_line)

                    except ValueError:
                        print(f"❌ Invalid values in {label_file}, line {i + 1}: {line}")
                        results["issues"].append(f"Invalid values in {label_file}, line {i + 1}")
                        file_has_errors = True

                # Write fixed file if needed
                if fix and file_has_errors and fixed_lines:
                    with open(label_file, 'w') as f:
                        for line in fixed_lines:
                            f.write(line + '\n')
                    print(f"✅ Fixed {label_file}")

                if file_has_valid_labels:
                    results["valid_files"] += 1
                else:
                    results["invalid_files"] += 1
                    invalid_files.append(label_file)

            except Exception as e:
                print(f"❌ Error processing {label_file}: {e}")
                results["issues"].append(f"Error processing {label_file}: {e}")
                results["invalid_files"] += 1

        # Print statistics for this split
        print(f"\n--- {split.upper()} Split ---")
        print(f"Label files: {len(label_files)}")
        print(f"Invalid files: {len(invalid_files)}")
        print(f"Empty files: {len(empty_files)}")

        # Print class distribution for this split
        if class_dist_by_split[split]:
            print("\nClass distribution:")
            for class_id, count in sorted(class_dist_by_split[split].items()):
                print(f"  Class {class_id}: {count} instances")

    # Calculate overall class distribution
    for split, dist in class_dist_by_split.items():
        for class_id, count in dist.items():
            if class_id not in results["class_distribution"]:
                results["class_distribution"][class_id] = 0
            results["class_distribution"][class_id] += count

    # Check for missing splits
    missing_splits = []
    for split in ['train', 'val']:
        if split not in file_counts_by_split or file_counts_by_split[split] == 0:
            missing_splits.append(split)

    if missing_splits:
        for split in missing_splits:
            print(f"❌ Missing {split} split (required for training)")
            results["issues"].append(f"Missing {split} split (required for training)")

    # Check for class issues
    if results["class_ids"]:
        max_class_id = max(results["class_ids"])
        needed_classes = max_class_id + 1

        # Get expected classes from dataset.yaml
        yaml_file = dataset_path / "dataset.yaml"
        yaml_nc = 0
        if yaml_file.exists():
            try:
                with open(yaml_file, 'r') as f:
                    yaml_content = yaml.safe_load(f)
                    if 'nc' in yaml_content:
                        yaml_nc = yaml_content['nc']
            except Exception:
                pass

        if needed_classes > yaml_nc:
            print(f"❌ Number of classes mismatch: Found class ID {max_class_id} but nc={yaml_nc} in dataset.yaml")
            results["issues"].append(f"Number of classes mismatch: need {needed_classes} classes, but have {yaml_nc}")

    # Create visualizations
    if visualize and results["bbox_sizes"]:
        try:
            output_dir = dataset_path / "analysis"
            output_dir.mkdir(exist_ok=True)

            # Class distribution bar chart
            plt.figure(figsize=(10, 6))
            classes = sorted(results["class_distribution"].keys())
            counts = [results["class_distribution"][c] for c in classes]
            plt.bar(classes, counts)
            plt.xlabel('Class ID')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(classes)
            plt.savefig(output_dir / "class_distribution.png")

            # Bounding box size distribution
            plt.figure(figsize=(10, 6))
            plt.hist(results["bbox_sizes"], bins=50, alpha=0.7)
            plt.xlabel('Bounding Box Area (relative to image)')
            plt.ylabel('Count')
            plt.title('Bounding Box Size Distribution')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(output_dir / "bbox_size_distribution.png")

            # Aspect ratio distribution
            plt.figure(figsize=(10, 6))
            plt.hist(results["aspect_ratios"], bins=50, alpha=0.7)
            plt.xlabel('Aspect Ratio (width/height)')
            plt.ylabel('Count')
            plt.title('Bounding Box Aspect Ratio Distribution')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(output_dir / "aspect_ratio_distribution.png")

            print(f"\nVisualization images saved to {output_dir}")
        except Exception as e:
            print(f"Error creating visualizations: {e}")

    # Print overall summary
    print("\n--- SUMMARY ---")
    print(f"Total label files: {results['total_files']}")
    print(f"Valid files: {results['valid_files']}")
    print(f"Invalid files: {results['invalid_files']}")
    print(f"Empty files: {results['empty_files']}")

    if results["class_distribution"]:
        print("\nOverall class distribution:")
        for class_id, count in sorted(results["class_distribution"].items()):
            print(f"  Class {class_id}: {count} instances")

    if results["issues"]:
        print(f"\nFound {len(results['issues'])} issues")
        if fix:
            print(f"Applied {len(results['fixes'])} fixes")
    else:
        print("\n✅ No issues found in label files")

    return results


def update_dataset_yaml(dataset_dir="dataset", class_ids=None):
    """
    Update dataset.yaml file based on analysis results.

    Args:
        dataset_dir: Path to dataset directory
        class_ids: Set of class IDs found in labels
    """
    if not class_ids:
        print("No class IDs provided for updating dataset.yaml")
        return False

    dataset_path = Path(dataset_dir)
    yaml_file = dataset_path / "dataset.yaml"

    # Get existing content
    yaml_content = {}
    if yaml_file.exists():
        try:
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
        except Exception as e:
            print(f"Error reading dataset.yaml: {e}")
            return False

    # Get class names from classes.txt if available
    class_names = []
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        try:
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading classes.txt: {e}")

    # Ensure we have enough class names
    max_class_id = max(class_ids) if class_ids else -1
    num_classes = max_class_id + 1

    # Extend class names if needed
    if len(class_names) < num_classes:
        for i in range(len(class_names), num_classes):
            class_names.append(f"class_{i}")

    # Update YAML content
    yaml_content['path'] = str(dataset_path.absolute())
    if 'train' not in yaml_content:
        yaml_content['train'] = 'images/train'
    if 'val' not in yaml_content:
        yaml_content['val'] = 'images/val'

    yaml_content['nc'] = num_classes
    yaml_content['names'] = class_names[:num_classes]  # Ensure correct length

    # Save updated YAML
    try:
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print(f"✅ Updated dataset.yaml with {num_classes} classes")
        return True
    except Exception as e:
        print(f"Error writing dataset.yaml: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Statistics and Fix Tool for YOLOv5 Training")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--fix", action="store_true", help="Fix label issues automatically")
    parser.add_argument("--visualize", action="store_true", help="Create visualization plots")
    parser.add_argument("--update-yaml", action="store_true", help="Update dataset.yaml based on analysis")

    args = parser.parse_args()

    # Analyze labels
    results = analyze_labels(args.dataset, args.fix, args.visualize)

    # Update dataset.yaml if requested
    if args.update_yaml and results["class_ids"]:
        update_dataset_yaml(args.dataset, results["class_ids"])
        print("\nRun training again with the updated dataset.yaml configuration")