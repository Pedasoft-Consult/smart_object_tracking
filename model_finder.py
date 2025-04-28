#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Finder - Script to locate trained model files
This script helps find model files that might be in unexpected locations
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path


def find_model_files(base_dir, training_id=None, hours=24):
    """Find model files in the specified directory"""
    base_dir = Path(base_dir)
    print(f"Searching for model files in: {base_dir}")

    # Calculate cutoff time
    cutoff_time = time.time() - (hours * 3600)

    # Find all .pt files
    pt_files = []

    # Method 1: Recursive glob
    for pt_file in base_dir.glob("**/*.pt"):
        if pt_file.is_file():
            # Filter by training ID if specified
            if training_id and training_id not in str(pt_file):
                continue

            mod_time = pt_file.stat().st_mtime
            if mod_time >= cutoff_time:
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                pt_files.append((pt_file, mod_time, size_mb))

    # Sort by modification time (newest first)
    pt_files.sort(key=lambda x: x[1], reverse=True)

    # Print results
    if pt_files:
        print(f"\nFound {len(pt_files)} model files:")
        for i, (file, mod_time, size) in enumerate(pt_files):
            print(f"  {i + 1}. {file} ({size:.2f} MB)")
            print(f"     Modified: {time.ctime(mod_time)}")
    else:
        print("\nNo model files found!")

    return pt_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find trained model files")
    parser.add_argument("base_dir", help="Base directory to search from")
    parser.add_argument("--training-id", help="Training ID to filter results")
    parser.add_argument("--hours", type=int, default=24,
                        help="Only include files modified in the last X hours")

    args = parser.parse_args()
    find_model_files(args.base_dir, args.training_id, args.hours)