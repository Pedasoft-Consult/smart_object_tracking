#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check Dataset YAML Configuration for YOLOv5 Training.

This script inspects your dataset.yaml file and compares it with
the actual classes found in your label files to identify mismatches.
"""

import os
import yaml
import argparse
from pathlib import Path
from collections import Counter


def inspect_dataset_yaml(dataset_dir="dataset", verbose=True):
    """
    Inspect dataset.yaml and compare with actual label files.

    Args:
        dataset_dir: Path to dataset directory
        verbose: Whether to print detailed information

    Returns:
        dict: Analysis results
    """
    dataset_path = Path(dataset_dir)
    yaml_file = dataset_path / "dataset.yaml"

    results = {
        "yaml_exists": False,
        "yaml_content": None,
        "class_counts": {},
        "max_class_id": -1,
        "unique_class_ids": set(),
        "issues": [],
        "recommendations": []
    }

    # Check if dataset.yaml exists
    if not yaml_file.exists():
        results["issues"].append("dataset.yaml file not found")
        results["recommendations"].append("Run the fix_dataset_classes.py script to create dataset.yaml")
        if verbose:
            print("❌ dataset.yaml file not found")
        return results

    # Read dataset.yaml
    try:
        with open(yaml_file, 'r') as f:
            yaml_content = yaml.safe_load(f)
        results["yaml_exists"] = True
        results["yaml_content"] = yaml_content

        if verbose:
            print(f"✅ dataset.yaml found with {yaml_content.get('nc', 0)} classes")
            print("\nYAML Content:")
            for key, value in yaml_content.items():
                print(f"  {key}: {value}")
    except Exception as e:
        results["issues"].append(f"Error reading dataset.yaml: {e}")
        if verbose:
            print(f"❌ Error reading dataset.yaml: {e}")
        return results

    # Check if required fields are present
    required_fields = ['path', 'train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in yaml_content:
            results["issues"].append(f"Missing required field in dataset.yaml: {field}")
            if verbose:
                print(f"❌ Missing required field: {field}")

    # Check for class count mismatch
    if 'nc' in yaml_content and 'names' in yaml_content:
        if yaml_content['nc'] != len(yaml_content['names']):
            results["issues"].append(
                f"Class count mismatch: nc={yaml_content['nc']} but names list has {len(yaml_content['names'])} entries"
            )
            if verbose:
                print(
                    f"❌ Class count mismatch: nc={yaml_content['nc']} but names list has {len(yaml_content['names'])} entries")

    # Analyze label files to find actual class IDs
    labels_path = dataset_path / "labels"
    if not labels_path.exists():
        results["issues"].append("Labels directory not found")
        if verbose:
            print("❌ Labels directory not found")
        return results

    # Count class IDs in label files
    class_counts = Counter()
    label_count = 0

    for split in ['train', 'val', 'test']:
        split_dir = labels_path / split
        if not split_dir.exists():
            continue

        label_files = list(split_dir.glob('*.txt'))

        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) >= 5:  # YOLO format requires at least 5 values
                            try:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                                label_count += 1
                                results["unique_class_ids"].add(class_id)
                            except ValueError:
                                pass
            except Exception:
                pass

    results["class_counts"] = dict(class_counts)

    if results["unique_class_ids"]:
        results["max_class_id"] = max(results["unique_class_ids"])

    # Check for class ID issues
    if 'nc' in yaml_content:
        yaml_nc = yaml_content['nc']
        max_class_id = results["max_class_id"]

        if max_class_id >= yaml_nc:
            results["issues"].append(
                f"Class ID too high: Found class ID {max_class_id} but nc={yaml_nc} in dataset.yaml"
            )
            results["recommendations"].append(
                f"Increase nc to at least {max_class_id + 1} in dataset.yaml or remap your class IDs"
            )
            if verbose:
                print(f"\n❌ Class ID too high: Found class ID {max_class_id} but nc={yaml_nc} in dataset.yaml")
                print(f"   Update dataset.yaml to have nc={max_class_id + 1} or higher")

    # Print summary of classes found
    if verbose and class_counts:
        print("\nClass distribution in label files:")
        for class_id, count in sorted(class_counts.items()):
            class_name = "Unknown"
            if ('names' in yaml_content and
                    yaml_content['names'] and
                    0 <= class_id < len(yaml_content['names'])):
                class_name = yaml_content['names'][class_id]

            print(f"  Class {class_id} ({class_name}): {count} instances")

    # Generate recommendations
    if not results["issues"]:
        if verbose:
            print("\n✅ No issues found in dataset.yaml")
    else:
        results["recommendations"].append("Run the fix_dataset_classes.py script to fix class issues")
        if verbose:
            print("\nRecommended actions:")
            for rec in results["recommendations"]:
                print(f"  - {rec}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Dataset YAML for YOLOv5 Training")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    results = inspect_dataset_yaml(args.dataset, verbose=not args.quiet)

    if args.quiet and results["issues"]:
        print(f"Found {len(results['issues'])} issues in dataset.yaml")
        for issue in results["issues"]:
            print(f"- {issue}")