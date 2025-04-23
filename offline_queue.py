#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Offline queue for tracking detections.
Stores detection results when offline for later upload.
"""

import os
import json
import time
import logging
import threading
import queue
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class OfflineQueue:
    """Queue for storing detections when offline"""

    def __init__(self, queue_dir, max_size=1000):
        """
        Initialize offline queue

        Args:
            queue_dir: Directory to store queue files
            max_size: Maximum number of items in memory queue
        """
        self.queue_dir = Path(queue_dir)
        self.max_size = max_size
        self.queue = queue.Queue(maxsize=max_size)
        self.logger = logging.getLogger('OfflineQueue')

        # Create queue directory if it doesn't exist
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Create data directory for images
        self.data_dir = self.queue_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Set up metadata file
        self.metadata_file = self.queue_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # Set up disk dump thread
        self.stop_event = threading.Event()
        self.dump_thread = threading.Thread(target=self._disk_dump_worker, daemon=True)
        self.dump_thread.start()

        self.logger.info(f"Offline queue initialized at {self.queue_dir}")
        self.logger.info(f"Queue contains {len(self.metadata['items'])} pending items")

    def _load_metadata(self):
        """
        Load queue metadata

        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}, creating new file")

        # Create default metadata
        return {
            "items": [],
            "last_update": time.time(),
            "queue_version": 1
        }

    def _save_metadata(self):
        """Save queue metadata"""
        try:
            self.metadata["last_update"] = time.time()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def _disk_dump_worker(self):
        """Background worker to dump queue items to disk"""
        while not self.stop_event.is_set():
            try:
                # Get item from queue without blocking
                try:
                    item = self.queue.get(block=False)
                except queue.Empty:
                    # Sleep and try again
                    time.sleep(0.5)
                    continue

                # Process item
                self._save_item_to_disk(item)

                # Mark task as done
                self.queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in disk dump worker: {e}")
                # Sleep to avoid tight loop on error
                time.sleep(1)

    def _save_item_to_disk(self, item):
        """
        Save queue item to disk

        Args:
            item: Queue item dictionary
        """
        try:
            timestamp = time.time()
            item_id = f"{timestamp:.6f}_{item['frame_id']}"

            # Save image if present
            if 'image' in item and item['image'] is not None:
                image_path = self.data_dir / f"{item_id}.jpg"
                cv2.imwrite(str(image_path), item['image'])

                # Replace image with file path
                item['image_path'] = str(image_path.relative_to(self.queue_dir))
                del item['image']

            # Add item metadata
            item['timestamp'] = timestamp
            item['id'] = item_id
            item['uploaded'] = False

            # Add to metadata
            self.metadata['items'].append(item)
            self._save_metadata()

            self.logger.debug(f"Saved item {item_id} to disk")

        except Exception as e:
            self.logger.error(f"Error saving item to disk: {e}")

    def add(self, frame_id, detections, image=None, metadata=None):
        """
        Add detection to queue

        Args:
            frame_id: Frame identifier
            detections: List of detections
            image: Optional frame image
            metadata: Optional additional metadata

        Returns:
            True if added successfully, False otherwise
        """
        try:
            # Create item
            item = {
                "frame_id": frame_id,
                "detections": detections.tolist() if isinstance(detections, np.ndarray) else detections,
                "timestamp": time.time()
            }

            # Add image if provided
            if image is not None:
                item['image'] = image.copy()

            # Add additional metadata
            if metadata:
                item['metadata'] = metadata

            # Add to queue
            try:
                self.queue.put(item, block=False)
                return True
            except queue.Full:
                self.logger.warning("Queue full, dropping item")
                return False

        except Exception as e:
            self.logger.error(f"Error adding item to queue: {e}")
            return False

    def get_pending_items(self, limit=100):
        """
        Get pending items that haven't been uploaded

        Args:
            limit: Maximum number of items to return

        Returns:
            List of pending items
        """
        return [item for item in self.metadata['items']
                if not item.get('uploaded', False)][:limit]

    def mark_as_uploaded(self, item_ids):
        """
        Mark items as uploaded

        Args:
            item_ids: List of item IDs

        Returns:
            Number of items marked
        """
        count = 0
        for item in self.metadata['items']:
            if item.get('id') in item_ids:
                item['uploaded'] = True
                count += 1

        # Save metadata if items were marked
        if count > 0:
            self._save_metadata()
            self.logger.info(f"Marked {count} items as uploaded")

        return count

    def cleanup(self, max_age_days=7, keep_uploaded=False):
        """
        Clean up old queue items

        Args:
            max_age_days: Maximum age of items to keep
            keep_uploaded: Whether to keep uploaded items

        Returns:
            Number of items removed
        """
        try:
            # Calculate cutoff time
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

            # Filter items
            old_count = len(self.metadata['items'])
            self.metadata['items'] = [
                item for item in self.metadata['items']
                if (item.get('timestamp', 0) > cutoff_time or
                    (keep_uploaded and item.get('uploaded', False)))
            ]

            # Calculate number of removed items
            removed_count = old_count - len(self.metadata['items'])

            if removed_count > 0:
                self._save_metadata()
                self.logger.info(f"Removed {removed_count} old items from queue")

                # Cleanup orphaned data files
                self._cleanup_data_files()

            return removed_count

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0

    def _cleanup_data_files(self):
        """Clean up orphaned data files"""
        try:
            # Get all image files
            image_files = list(self.data_dir.glob("*.jpg"))

            # Get all referenced image paths
            referenced_paths = set()
            for item in self.metadata['items']:
                if 'image_path' in item:
                    referenced_paths.add(self.queue_dir / item['image_path'])

            # Delete orphaned files
            deleted_count = 0
            for image_file in image_files:
                if image_file not in referenced_paths:
                    image_file.unlink()
                    deleted_count += 1

            if deleted_count > 0:
                self.logger.info(f"Deleted {deleted_count} orphaned image files")

        except Exception as e:
            self.logger.error(f"Error cleaning up data files: {e}")

    def stats(self):
        """
        Get queue statistics

        Returns:
            Dictionary with queue statistics
        """
        total_items = len(self.metadata['items'])
        uploaded_items = sum(1 for item in self.metadata['items'] if item.get('uploaded', False))
        pending_items = total_items - uploaded_items

        # Calculate queue size on disk
        queue_size = 0
        try:
            if self.metadata_file.exists():
                queue_size += self.metadata_file.stat().st_size

            for file in self.data_dir.glob("*"):
                queue_size += file.stat().st_size
        except Exception:
            pass

        return {
            "total_items": total_items,
            "uploaded_items": uploaded_items,
            "pending_items": pending_items,
            "queue_size_bytes": queue_size,
            "queue_size_mb": queue_size / (1024 * 1024),
            "in_memory_items": self.queue.qsize()
        }

    def flush(self):
        """
        Flush memory queue to disk

        Returns:
            Number of items flushed
        """
        count = 0
        while not self.queue.empty():
            try:
                item = self.queue.get(block=False)
                self._save_item_to_disk(item)
                self.queue.task_done()
                count += 1
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error flushing queue: {e}")
                break

        return count

    def shutdown(self):
        """Shutdown queue and flush to disk"""
        self.logger.info("Shutting down offline queue")

        # Stop background thread
        self.stop_event.set()
        if self.dump_thread.is_alive():
            self.dump_thread.join(timeout=5.0)

        # Flush remaining items
        flushed = self.flush()
        self.logger.info(f"Flushed {flushed} items to disk during shutdown")

        # Final save of metadata
        self._save_metadata()