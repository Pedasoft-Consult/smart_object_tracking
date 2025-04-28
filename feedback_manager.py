#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feedback Manager for Smart Object Tracking System.
Collects and processes user feedback for model improvement.
"""

import os
import json
import time
import logging
import cv2
import numpy as np
from pathlib import Path
import threading
import queue


class FeedbackManager:
    """
    Manages feedback collection and processing for model improvement.
    Collects corrected detections and feeds them back into the training pipeline.
    """

    def __init__(self, feedback_dir, dataset_manager=None):
        """
        Initialize feedback manager.

        Args:
            feedback_dir: Directory to store feedback data
            dataset_manager: Dataset manager instance for adding feedback to dataset
        """
        self.feedback_dir = Path(feedback_dir)
        self.dataset_manager = dataset_manager
        self.metadata_file = self.feedback_dir / "feedback_metadata.json"
        self.logger = logging.getLogger('FeedbackManager')

        # Create feedback processing queue
        self.feedback_queue = queue.Queue()
        self.process_thread = None
        self.stop_event = threading.Event()

        # Initialize directory and metadata
        self._initialize()

        # Start background processing thread
        self._start_processing_thread()

        self.logger.info(f"Feedback manager initialized at {self.feedback_dir}")

    def _initialize(self):
        """Initialize feedback directory and metadata"""
        # Create directory if it doesn't exist
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        (self.feedback_dir / "images").mkdir(exist_ok=True)

        # Load metadata if exists, otherwise create default
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
            "items": [],
            "last_update": time.time(),
            "version": 1,
            "stats": {
                "total_items": 0,
                "processed_items": 0,
                "pending_items": 0,
                "added_to_dataset": 0
            },
            "retraining_events": []
        }
        self._save_metadata()

    def _save_metadata(self):
        """Save feedback metadata to file"""
        try:
            self.metadata["last_update"] = time.time()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def _start_processing_thread(self):
        """Start background processing thread"""
        if self.process_thread is not None and self.process_thread.is_alive():
            return  # Thread already running

        self.stop_event.clear()
        self.process_thread = threading.Thread(target=self._process_feedback_worker, daemon=True)
        self.process_thread.start()
        self.logger.info("Started feedback processing thread")

    def _process_feedback_worker(self):
        """Background worker for processing feedback"""
        while not self.stop_event.is_set():
            try:
                # Try to get item from queue
                try:
                    item = self.feedback_queue.get(timeout=5.0)
                except queue.Empty:
                    continue  # No items, continue loop

                # Process feedback item
                self._process_feedback_item(item)

                # Mark task as done
                self.feedback_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in feedback processing thread: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                time.sleep(1)  # Sleep to avoid tight loop on error

    def _process_feedback_item(self, item):
        """
        Process a single feedback item.

        Args:
            item: Feedback item dictionary
        """
        try:
            # Check if already processed
            if item.get("processed", False):
                return

            # Check if we have a dataset manager
            if self.dataset_manager is None:
                self.logger.warning(f"Dataset manager not available, cannot process feedback item {item['id']}")
                return

            # Load image if available
            image = None
            if "image_path" in item and os.path.exists(item["image_path"]):
                image = cv2.imread(item["image_path"])

            if image is None:
                self.logger.warning(f"Image not found for feedback item {item['id']}")
                item["processed"] = True
                item["processing_error"] = "Image not found"
                self._update_item(item)
                return

            # Use corrected detections
            if "corrected_detections" not in item or not item["corrected_detections"]:
                self.logger.warning(f"No corrected detections for feedback item {item['id']}")
                item["processed"] = True
                item["processing_error"] = "No corrected detections"
                self._update_item(item)
                return

            # Add to dataset with appropriate split
            # Determine split based on configuration or random assignment
            import random
            splits = ["train", "val", "test"]
            weights = [0.8, 0.1, 0.1]  # Default split ratio
            split = random.choices(splits, weights=weights, k=1)[0]

            # Add to dataset
            image_id = self.dataset_manager.add_from_feedback(
                image=image,
                annotations=item["corrected_detections"],
                split=split,
                image_id=f"feedback_{item['id']}"
            )

            if image_id:
                item["processed"] = True
                item["added_to_dataset"] = True
                item["dataset_image_id"] = image_id
                item["dataset_split"] = split
                item["processing_time"] = time.time()

                # Update statistics
                self.metadata["stats"]["processed_items"] += 1
                self.metadata["stats"]["added_to_dataset"] += 1
                self.metadata["stats"]["pending_items"] -= 1

                self.logger.info(f"Feedback item {item['id']} added to dataset as {image_id} in {split} split")
            else:
                item["processing_error"] = "Failed to add to dataset"
                self.logger.warning(f"Failed to add feedback item {item['id']} to dataset")

            # Update item in metadata
            self._update_item(item)

        except Exception as e:
            self.logger.error(f"Error processing feedback item {item.get('id', 'unknown')}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

            # Update item with error
            item["processing_error"] = str(e)
            self._update_item(item)

    def _update_item(self, item):
        """
        Update an item in the metadata.

        Args:
            item: Updated item dictionary
        """
        # Find and update the item
        for i, existing_item in enumerate(self.metadata["items"]):
            if existing_item["id"] == item["id"]:
                self.metadata["items"][i] = item
                break

        # Save metadata
        self._save_metadata()

    def add_feedback(self, frame_id, original_detections, corrected_detections, image=None, metadata=None):
        """
        Add feedback item with corrected detections.

        Args:
            frame_id: Frame identifier
            original_detections: Original detection results
            corrected_detections: User-corrected detection results
            image: Frame image as numpy array
            metadata: Additional metadata for the feedback

        Returns:
            str: Feedback item ID
        """
        try:
            # Generate unique ID
            feedback_id = f"{time.time():.6f}_{frame_id}"

            # Create feedback item
            item = {
                "id": feedback_id,
                "frame_id": frame_id,
                "timestamp": time.time(),
                "original_detections": original_detections,
                "corrected_detections": corrected_detections,
                "processed": False
            }

            # Add additional metadata if provided
            if metadata:
                item["metadata"] = metadata

            # Save image if provided
            if image is not None:
                image_path = self.feedback_dir / "images" / f"{feedback_id}.jpg"
                cv2.imwrite(str(image_path), image)
                item["image_path"] = str(image_path)

            # Add to metadata
            self.metadata["items"].append(item)

            # Update statistics
            self.metadata["stats"]["total_items"] += 1
            self.metadata["stats"]["pending_items"] += 1

            self._save_metadata()

            # Add to processing queue
            self.feedback_queue.put(item)

            self.logger.info(f"Added feedback item {feedback_id} to queue")
            return feedback_id

        except Exception as e:
            self.logger.error(f"Error adding feedback: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def get_feedback_item(self, feedback_id):
        """
        Get feedback item by ID.

        Args:
            feedback_id: Feedback item ID

        Returns:
            dict: Feedback item or None if not found
        """
        for item in self.metadata["items"]:
            if item["id"] == feedback_id:
                return item
        return None

    def list_feedback_items(self, limit=100, offset=0, processed=None):
        """
        List feedback items.

        Args:
            limit: Maximum number of items to return
            offset: Offset for pagination
            processed: Filter by processed status (None for all)

        Returns:
            list: List of feedback items
        """
        # Filter by processed status if requested
        if processed is not None:
            filtered_items = [item for item in self.metadata["items"] if item.get("processed", False) == processed]
        else:
            filtered_items = self.metadata["items"]

        # Sort by timestamp (newest first)
        sorted_items = sorted(filtered_items, key=lambda x: x.get("timestamp", 0), reverse=True)

        # Apply pagination
        paginated_items = sorted_items[offset:offset + limit]

        return paginated_items

    def process_pending_feedback(self, limit=None):
        """
        Process pending feedback items.

        Args:
            limit: Maximum number of items to process (None for all)

        Returns:
            int: Number of items processed
        """
        # Get unprocessed items
        unprocessed = [item for item in self.metadata["items"] if not item.get("processed", False)]

        # Apply limit if specified
        if limit is not None:
            unprocessed = unprocessed[:limit]

        count = 0
        for item in unprocessed:
            # Add to processing queue
            self.feedback_queue.put(item)
            count += 1

        self.logger.info(f"Added {count} pending feedback items to processing queue")
        return count

    def trigger_retraining(self, model_trainer, min_feedback_items=100):
        """
        Check if enough feedback to retrain and trigger training.

        Args:
            model_trainer: Model trainer instance
            min_feedback_items: Minimum number of processed feedback items to trigger retraining

        Returns:
            str: Training ID or None if not triggered
        """
        # Check if we have enough processed feedback
        processed_count = self.metadata["stats"].get("processed_items", 0)

        if processed_count < min_feedback_items:
            self.logger.info(
                f"Not enough processed feedback items for retraining: {processed_count}/{min_feedback_items}")
            return None

        # Get last retraining timestamp
        last_retraining = 0
        if self.metadata.get("retraining_events"):
            last_retraining = max(event.get("timestamp", 0) for event in self.metadata["retraining_events"])

        # Check if new feedback since last retraining
        new_feedback = False
        for item in self.metadata["items"]:
            if item.get("processed", False) and item.get("processing_time", 0) > last_retraining:
                new_feedback = True
                break

        if not new_feedback:
            self.logger.info("No new feedback since last retraining")
            return None

        # Prepare training configuration
        training_config = model_trainer.prepare_training_config(
            model_type='yolov5s',  # Use default or from config
            epochs=50,  # Can be configured
            batch_size=16,
            img_size=640
        )

        if training_config is None:
            self.logger.error("Failed to prepare training configuration")
            return None

        # Start training
        training_id = model_trainer.train(training_config=training_config)

        if training_id:
            # Record retraining event
            self.metadata["retraining_events"].append({
                "timestamp": time.time(),
                "training_id": training_id,
                "feedback_count": processed_count,
                "triggered_by": "feedback_manager"
            })
            self._save_metadata()

            self.logger.info(f"Triggered retraining with {processed_count} feedback items, training ID: {training_id}")
            return training_id
        else:
            self.logger.error("Failed to start training")
            return None

    def get_statistics(self):
        """
        Get feedback statistics.

        Returns:
            dict: Feedback statistics
        """
        # Update statistics
        stats = self.metadata["stats"].copy()

        # Calculate detailed statistics
        total_items = len(self.metadata["items"])
        processed_items = sum(1 for item in self.metadata["items"] if item.get("processed", False))
        pending_items = total_items - processed_items
        added_to_dataset = sum(1 for item in self.metadata["items"]
                               if item.get("processed", False) and item.get("added_to_dataset", False))

        # Calculate correction statistics
        total_corrections = 0
        added_objects = 0
        removed_objects = 0
        modified_objects = 0

        for item in self.metadata["items"]:
            if not item.get("processed", False):
                continue

            original_count = len(item.get("original_detections", []))
            corrected_count = len(item.get("corrected_detections", []))

            total_corrections += 1
            if corrected_count > original_count:
                added_objects += 1
            elif corrected_count < original_count:
                removed_objects += 1
            else:
                modified_objects += 1

        # Update and return statistics
        stats.update({
            "total_items": total_items,
            "processed_items": processed_items,
            "pending_items": pending_items,
            "added_to_dataset": added_to_dataset,
            "correction_stats": {
                "total_corrections": total_corrections,
                "added_objects": added_objects,
                "removed_objects": removed_objects,
                "modified_objects": modified_objects
            },
            "retraining_count": len(self.metadata.get("retraining_events", []))
        })

        return stats

    def cleanup(self, max_age_days=30):
        """
        Clean up old feedback data.

        Args:
            max_age_days: Maximum age of feedback data to keep

        Returns:
            int: Number of items removed
        """
        try:
            # Calculate cutoff time
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

            # Count items before cleanup
            old_count = len(self.metadata["items"])

            # Filter items by age and processed status
            # Keep processed items that have been added to the dataset regardless of age
            new_items = [
                item for item in self.metadata["items"]
                if (item.get("timestamp", 0) > cutoff_time or
                    (item.get("processed", False) and item.get("added_to_dataset", False)))
            ]

            # Calculate removed count
            removed_count = old_count - len(new_items)

            if removed_count > 0:
                # Get IDs of removed items
                old_ids = {item["id"] for item in self.metadata["items"]}
                new_ids = {item["id"] for item in new_items}
                removed_ids = old_ids - new_ids

                # Remove associated image files
                for item_id in removed_ids:
                    image_path = self.feedback_dir / "images" / f"{item_id}.jpg"
                    if image_path.exists():
                        image_path.unlink()

                # Update metadata
                self.metadata["items"] = new_items

                # Update statistics
                processed_removed = sum(1 for item in self.metadata["items"]
                                        if item["id"] in removed_ids and item.get("processed", False))

                self.metadata["stats"]["total_items"] -= removed_count
                self.metadata["stats"]["processed_items"] -= processed_removed
                self.metadata["stats"]["pending_items"] -= (removed_count - processed_removed)

                self._save_metadata()

                self.logger.info(f"Removed {removed_count} old feedback items")

            return removed_count

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0

    def shutdown(self):
        """
        Shutdown feedback manager and wait for processing to complete.

        Returns:
            bool: Success status
        """
        try:
            # Signal processing thread to stop
            self.stop_event.set()

            # Wait for processing thread to complete
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=5.0)

            # Wait for queue to be processed
            if not self.feedback_queue.empty():
                self.logger.info(f"Waiting for {self.feedback_queue.qsize()} feedback items to be processed")
                try:
                    self.feedback_queue.join(timeout=10.0)
                except:
                    pass

            # Save final metadata
            self._save_metadata()

            self.logger.info("Feedback manager shut down")
            return True

        except Exception as e:
            self.logger.error(f"Error shutting down feedback manager: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize feedback manager
    feedback_manager = FeedbackManager("feedback")

    # Add test feedback
    feedback_id = feedback_manager.add_feedback(
        frame_id="test_frame",
        original_detections=[
            {"class_id": 0, "confidence": 0.8, "bbox": [100, 100, 200, 200]}
        ],
        corrected_detections=[
            {"class_id": 0, "confidence": 1.0, "bbox": [110, 110, 210, 210]},
            {"class_id": 1, "confidence": 0.9, "bbox": [300, 300, 400, 400]}
        ],
        # Create a test image (black with white rectangle)
        image=np.zeros((480, 640, 3), dtype=np.uint8)
    )

    print(f"Added feedback: {feedback_id}")

    # Print statistics
    stats = feedback_manager.get_statistics()
    print(f"Feedback statistics: {stats}")

    # Shutdown
    feedback_manager.shutdown()