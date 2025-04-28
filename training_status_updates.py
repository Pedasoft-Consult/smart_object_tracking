#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Status Updates Module for Smart Object Tracking System.
Provides functions to update and track training status and progress.
"""

import os
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger('TrainingStatusUpdates')


def load_metadata(metadata_file):
    """
    Load metadata from file.

    Args:
        metadata_file: Path to metadata file

    Returns:
        dict: Metadata dictionary
    """
    metadata_file = Path(metadata_file)
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}, creating new file")
            metadata = {
                "models": [],
                "training_runs": [],
                "last_updated": time.time()
            }
            save_metadata(metadata, metadata_file)
            return metadata
    else:
        metadata = {
            "models": [],
            "training_runs": [],
            "last_updated": time.time()
        }
        save_metadata(metadata, metadata_file)
        return metadata


def save_metadata(metadata, metadata_file):
    """
    Save metadata to file.

    Args:
        metadata: Metadata dictionary
        metadata_file: Path to metadata file
    """
    try:
        metadata["last_updated"] = time.time()
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")


def update_training_status(metadata_file, training_id, status, additional_info=None):
    """
    Update training status in metadata.

    Args:
        metadata_file: Path to metadata file
        training_id: Training ID
        status: Status string
        additional_info: Additional information dictionary

    Returns:
        bool: Success status
    """
    try:
        metadata = load_metadata(metadata_file)

        found = False
        for run in metadata["training_runs"]:
            if run.get("training_id") == training_id:
                run["status"] = status
                run["last_updated"] = time.time()

                if additional_info:
                    # Don't override existing fields with None values
                    for key, value in additional_info.items():
                        if value is not None or key not in run:
                            run[key] = value

                # Add timestamped status change
                if "status_history" not in run:
                    run["status_history"] = []

                run["status_history"].append({
                    "status": status,
                    "timestamp": time.time()
                })

                found = True
                break

        if not found:
            # Create a new training run entry
            new_run = {
                "training_id": training_id,
                "status": status,
                "last_updated": time.time(),
                "status_history": [{
                    "status": status,
                    "timestamp": time.time()
                }]
            }

            if additional_info:
                for key, value in additional_info.items():
                    if value is not None:
                        new_run[key] = value

            metadata["training_runs"].append(new_run)

        save_metadata(metadata, metadata_file)
        return True

    except Exception as e:
        logger.error(f"Error updating training status: {e}")
        return False


def update_training_progress(metadata_file, training_id, progress, metrics=None):
    """
    Update training progress in metadata.

    Args:
        metadata_file: Path to metadata file
        training_id: Training ID
        progress: Progress value (0.0 to 1.0)
        metrics: Dictionary of metrics

    Returns:
        bool: Success status
    """
    try:
        metadata = load_metadata(metadata_file)

        found = False
        for run in metadata["training_runs"]:
            if run.get("training_id") == training_id:
                run["progress"] = progress
                run["last_updated"] = time.time()

                if metrics:
                    if "metrics" not in run:
                        run["metrics"] = {}
                    run["metrics"].update(metrics)

                found = True
                break

        if not found:
            # Create a new training run entry with progress
            new_run = {
                "training_id": training_id,
                "status": "training",  # Default status for new progress entries
                "progress": progress,
                "last_updated": time.time(),
                "status_history": [{
                    "status": "training",
                    "timestamp": time.time()
                }]
            }

            if metrics:
                new_run["metrics"] = metrics

            metadata["training_runs"].append(new_run)

        save_metadata(metadata, metadata_file)
        return True

    except Exception as e:
        logger.error(f"Error updating training progress: {e}")
        return False


def get_training_status(metadata_file, training_id):
    """
    Get training status information.

    Args:
        metadata_file: Path to metadata file
        training_id: Training ID

    Returns:
        dict: Training status information
    """
    try:
        metadata = load_metadata(metadata_file)

        for run in metadata.get("training_runs", []):
            if run.get("training_id") == training_id:
                return {
                    "training_id": training_id,
                    "status": run.get("status", "unknown"),
                    "progress": run.get("progress", 0.0),
                    "metrics": run.get("metrics", {}),
                    "last_updated": run.get("last_updated", 0)
                }

        return {"status": "not_found", "training_id": training_id}

    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return {"status": "error", "error": str(e)}


def get_all_training_runs(metadata_file, limit=None, status=None):
    """
    Get list of all training runs.

    Args:
        metadata_file: Path to metadata file
        limit: Maximum number of runs to return (latest first)
        status: Filter by status (e.g., 'completed', 'failed')

    Returns:
        list: List of training run information
    """
    try:
        metadata = load_metadata(metadata_file)
        runs = metadata.get("training_runs", [])

        # Filter by status if requested
        if status is not None:
            runs = [run for run in runs if run.get("status") == status]

        # Sort by timestamp (newest first)
        runs = sorted(runs, key=lambda x: x.get("last_updated", 0), reverse=True)

        # Apply limit
        if limit is not None and isinstance(limit, int) and limit > 0:
            runs = runs[:limit]

        return runs

    except Exception as e:
        logger.error(f"Error getting training runs: {e}")
        return []


# Main function for command-line usage
if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training Status Updates")
    parser.add_argument('--metadata', type=str, required=True, help="Path to metadata file")
    parser.add_argument('--action', type=str, required=True,
                        choices=["update-status", "update-progress", "get-status", "list-runs"],
                        help="Action to perform")
    parser.add_argument('--training-id', type=str, help="Training ID")
    parser.add_argument('--status', type=str, help="Status to set")
    parser.add_argument('--progress', type=float, help="Progress value (0.0 to 1.0)")
    parser.add_argument('--metrics', type=str, help="Metrics JSON string")
    parser.add_argument('--limit', type=int, help="Limit for list-runs action")
    parser.add_argument('--filter', type=str, help="Status filter for list-runs action")

    args = parser.parse_args()

    # Perform requested action
    if args.action == "update-status":
        if args.training_id and args.status:
            additional_info = {}
            if args.metrics:
                try:
                    additional_info["metrics"] = json.loads(args.metrics)
                except json.JSONDecodeError:
                    logger.error("Invalid metrics JSON string")

            success = update_training_status(args.metadata, args.training_id, args.status, additional_info)
            print(f"Status update {'successful' if success else 'failed'}")
        else:
            print("Training ID and status required for update-status action")

    elif args.action == "update-progress":
        if args.training_id and args.progress is not None:
            metrics = None
            if args.metrics:
                try:
                    metrics = json.loads(args.metrics)
                except json.JSONDecodeError:
                    logger.error("Invalid metrics JSON string")

            success = update_training_progress(args.metadata, args.training_id, args.progress, metrics)
            print(f"Progress update {'successful' if success else 'failed'}")
        else:
            print("Training ID and progress required for update-progress action")

    elif args.action == "get-status":
        if args.training_id:
            status = get_training_status(args.metadata, args.training_id)
            print(json.dumps(status, indent=2))
        else:
            print("Training ID required for get-status action")

    elif args.action == "list-runs":
        runs = get_all_training_runs(args.metadata, args.limit, args.filter)
        print(f"Found {len(runs)} training runs:")
        for run in runs:
            print(json.dumps(run, indent=2))
            print("-" * 40)