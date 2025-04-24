#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Offline Queue for tracking detections.
Provides:
- Priority-based uploading
- Compression of queued data
- Partial uploads when connectivity is unstable
- Batched uploads to reduce API calls
- Progressive JPEG for more efficient image storage
"""

import os
import json
import time
import logging
import threading
import queue
import cv2
import numpy as np
import zlib
import io
from pathlib import Path
from datetime import datetime
import requests
from PIL import Image
import random
import hashlib


class EnhancedOfflineQueue:
    """Enhanced queue for storing detections when offline"""

    def __init__(self, queue_dir, max_size=1000, batch_size=20):
        """
        Initialize enhanced offline queue

        Args:
            queue_dir: Directory to store queue files
            max_size: Maximum number of items in memory queue
            batch_size: Number of items to upload in a batch
        """
        self.queue_dir = Path(queue_dir)
        self.max_size = max_size
        self.batch_size = batch_size
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.logger = logging.getLogger('EnhancedOfflineQueue')

        # Create queue directory if it doesn't exist
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # Create data directory for images
        self.data_dir = self.queue_dir / "data"
        self.data_dir.mkdir(exist_ok=True)

        # Set up metadata file
        self.metadata_file = self.queue_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # Upload status tracking
        self.upload_lock = threading.Lock()
        self.is_uploading = False
        self.upload_progress = {
            "total": 0,
            "uploaded": 0,
            "failed": 0,
            "in_progress": False,
            "last_upload": None,
            "last_error": None
        }

        # Set up disk dump thread
        self.stop_event = threading.Event()
        self.dump_thread = threading.Thread(target=self._disk_dump_worker, daemon=True)
        self.dump_thread.start()

        self.logger.info(f"Enhanced offline queue initialized at {self.queue_dir}")
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
            "queue_version": 2,  # Updated version
            "compression": True,
            "progressive_jpeg": True
        }

    def _save_metadata(self):
        """Save queue metadata"""
        try:
            self.metadata["last_update"] = time.time()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def _compress_data(self, data):
        """
        Compress data using zlib

        Args:
            data: Data to compress (dict or list)

        Returns:
            Compressed data as bytes
        """
        try:
            json_data = json.dumps(data).encode('utf-8')
            compressed = zlib.compress(json_data, level=9)  # Highest compression
            return compressed
        except Exception as e:
            self.logger.error(f"Error compressing data: {e}")
            # Return uncompressed JSON as fallback
            return json.dumps(data).encode('utf-8')

    def _decompress_data(self, compressed_data):
        """
        Decompress data using zlib

        Args:
            compressed_data: Compressed data bytes

        Returns:
            Original data (dict or list)
        """
        try:
            decompressed = zlib.decompress(compressed_data)
            return json.loads(decompressed.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"Error decompressing data: {e}")
            try:
                # Try to parse as uncompressed JSON
                return json.loads(compressed_data.decode('utf-8'))
            except:
                return None

    def _disk_dump_worker(self):
        """Background worker to dump queue items to disk"""
        while not self.stop_event.is_set():
            try:
                # Get item from queue without blocking
                try:
                    priority, item = self.queue.get(block=False)
                except queue.Empty:
                    # Sleep and try again
                    time.sleep(0.5)
                    continue

                # Process item
                self._save_item_to_disk(item, priority)

                # Mark task as done
                self.queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in disk dump worker: {e}")
                # Sleep to avoid tight loop on error
                time.sleep(1)

    def _save_item_to_disk(self, item, priority=100):
        """
        Save queue item to disk

        Args:
            item: Queue item dictionary
            priority: Item priority (lower = higher priority)
        """
        try:
            timestamp = time.time()
            item_id = f"{timestamp:.6f}_{item['frame_id']}"

            # Save image if present using progressive JPEG if enabled
            if 'image' in item and item['image'] is not None:
                image_path = self.data_dir / f"{item_id}.jpg"

                if self.metadata.get('progressive_jpeg', True):
                    # Use PIL for progressive JPEG
                    img = Image.fromarray(cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB))
                    img.save(str(image_path), format='JPEG', quality=85, progressive=True, optimize=True)
                else:
                    # Use OpenCV for regular JPEG
                    cv2.imwrite(str(image_path), item['image'], [cv2.IMWRITE_JPEG_QUALITY, 85])

                # Replace image with file path
                item['image_path'] = str(image_path.relative_to(self.queue_dir))
                del item['image']

            # Add item metadata
            item['timestamp'] = timestamp
            item['id'] = item_id
            item['uploaded'] = False
            item['priority'] = priority
            item['retry_count'] = 0
            item['size'] = len(json.dumps(item))  # Approximate size for sorting

            # Calculate checksum for integrity verification
            item['checksum'] = hashlib.md5(json.dumps(item, sort_keys=True).encode('utf-8')).hexdigest()

            # Add to metadata
            self.metadata['items'].append(item)
            self._save_metadata()

            self.logger.debug(f"Saved item {item_id} to disk with priority {priority}")

        except Exception as e:
            self.logger.error(f"Error saving item to disk: {e}")

    def add(self, frame_id, detections, image=None, metadata=None, priority=100):
        """
        Add detection to queue with priority

        Args:
            frame_id: Frame identifier
            detections: List of detections
            image: Optional frame image
            metadata: Optional additional metadata
            priority: Item priority (lower = higher priority)

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

            # Add to queue with priority
            try:
                self.queue.put((priority, item), block=False)
                return True
            except queue.Full:
                self.logger.warning("Queue full, dropping item")
                return False

        except Exception as e:
            self.logger.error(f"Error adding item to queue: {e}")
            return False

    def add_high_priority(self, frame_id, detections, image=None, metadata=None):
        """Add high priority detection to queue"""
        return self.add(frame_id, detections, image, metadata, priority=50)

    def add_critical_priority(self, frame_id, detections, image=None, metadata=None):
        """Add critical priority detection to queue (highest priority)"""
        return self.add(frame_id, detections, image, metadata, priority=10)

    def get_pending_items(self, limit=100, batch_size=None):
        """
        Get pending items that haven't been uploaded, ordered by priority

        Args:
            limit: Maximum number of items to return
            batch_size: Optional batch size to override instance setting

        Returns:
            List of pending items
        """
        # Use instance batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size

        # Limit to the smaller of limit or batch_size
        count = min(limit, batch_size)

        # Get all pending items
        pending = [item for item in self.metadata['items']
                   if not item.get('uploaded', False)]

        # Sort by priority (lower number = higher priority)
        pending.sort(key=lambda x: (x.get('priority', 100), x.get('retry_count', 0)))

        return pending[:count]

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
                item['upload_time'] = time.time()
                count += 1

        # Save metadata if items were marked
        if count > 0:
            self._save_metadata()
            self.logger.info(f"Marked {count} items as uploaded")

        return count

    def mark_failed_upload(self, item_ids, error=None):
        """
        Mark items as failed upload and increment retry count

        Args:
            item_ids: List of item IDs
            error: Optional error message

        Returns:
            Number of items marked
        """
        count = 0
        for item in self.metadata['items']:
            if item.get('id') in item_ids:
                item['last_error'] = error
                item['last_attempt'] = time.time()
                item['retry_count'] = item.get('retry_count', 0) + 1
                count += 1

        # Save metadata if items were marked
        if count > 0:
            self._save_metadata()
            self.logger.info(f"Marked {count} items as failed upload")

        return count

    def upload_pending_items(self, url, headers=None, max_retries=3, timeout=30):
        """
        Upload pending items to server in batches

        Args:
            url: Server URL to upload items to
            headers: Optional request headers
            max_retries: Maximum number of retry attempts per batch
            timeout: Request timeout in seconds

        Returns:
            Dictionary with upload statistics
        """
        # Prevent multiple upload operations at the same time
        with self.upload_lock:
            if self.is_uploading:
                return {"error": "Upload already in progress"}
            self.is_uploading = True
            self.upload_progress = {
                "total": 0,
                "uploaded": 0,
                "failed": 0,
                "in_progress": True,
                "start_time": time.time(),
                "last_error": None
            }

        try:
            # Get pending items count
            pending = [item for item in self.metadata['items']
                       if not item.get('uploaded', False)]

            total_items = len(pending)
            if total_items == 0:
                self.logger.info("No pending items to upload")
                self.is_uploading = False
                self.upload_progress["in_progress"] = False
                return self.upload_progress

            self.upload_progress["total"] = total_items

            # Default headers
            if headers is None:
                headers = {'Content-Type': 'application/octet-stream'}

            # Process items in batches
            batch_start = 0
            while batch_start < total_items and self.is_uploading:
                # Get next batch
                batch = self.get_pending_items(limit=self.batch_size)
                if not batch:
                    break  # No more items

                batch_ids = [item['id'] for item in batch]

                # Try to upload the batch
                uploaded = False
                retries = 0
                while not uploaded and retries < max_retries and self.is_uploading:
                    try:
                        # Compress batch data
                        batch_data = self._compress_data(batch)

                        # Upload batch
                        response = requests.post(
                            url,
                            data=batch_data,
                            headers=headers,
                            timeout=timeout
                        )

                        # Check response
                        if response.status_code == 200:
                            # Mark as uploaded
                            self.mark_as_uploaded(batch_ids)
                            self.upload_progress["uploaded"] += len(batch)
                            uploaded = True
                            self.logger.info(f"Uploaded batch of {len(batch)} items")
                        else:
                            # Failed upload
                            error = f"Server returned {response.status_code}: {response.text}"
                            self.mark_failed_upload(batch_ids, error)
                            self.upload_progress["last_error"] = error
                            self.logger.warning(f"Failed to upload batch: {error}")
                            retries += 1
                            time.sleep(1)  # Wait before retry

                    except Exception as e:
                        # Error during upload
                        self.mark_failed_upload(batch_ids, str(e))
                        self.upload_progress["last_error"] = str(e)
                        self.logger.error(f"Error uploading batch: {e}")
                        retries += 1
                        time.sleep(1)  # Wait before retry

                # If we failed after all retries
                if not uploaded:
                    self.upload_progress["failed"] += len(batch)

                # Move to next batch
                batch_start += self.batch_size

            return self.upload_progress

        finally:
            # Update progress
            self.upload_progress["in_progress"] = False
            self.upload_progress["end_time"] = time.time()
            self.upload_progress["duration"] = self.upload_progress["end_time"] - self.upload_progress["start_time"]
            self.is_uploading = False

    def cancel_upload(self):
        """Cancel current upload operation"""
        if self.is_uploading:
            self.is_uploading = False
            self.logger.info("Upload operation cancelled")
            return True
        return False

    def cleanup(self, max_age_days=7, keep_uploaded=False, max_size_mb=None):
        """
        Clean up old queue items

        Args:
            max_age_days: Maximum age of items to keep
            keep_uploaded: Whether to keep uploaded items
            max_size_mb: Maximum queue size in MB

        Returns:
            Number of items removed
        """
        try:
            # Calculate cutoff time
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

            # Filter items by age
            old_count = len(self.metadata['items'])
            self.metadata['items'] = [
                item for item in self.metadata['items']
                if (item.get('timestamp', 0) > cutoff_time or
                    (keep_uploaded and item.get('uploaded', False)))
            ]

            # Calculate number of removed items
            removed_count = old_count - len(self.metadata['items'])

            # If max_size_mb is specified, also remove oldest items exceeding the size
            if max_size_mb is not None:
                # Calculate current queue size
                queue_size = self._calculate_queue_size() / (1024 * 1024)  # Convert to MB

                if queue_size > max_size_mb:
                    # Sort by priority, uploaded status, and then age
                    self.metadata['items'].sort(key=lambda x: (
                        0 if x.get('uploaded', False) else 1,  # Uploaded items first to keep
                        x.get('priority', 100),  # Higher priority items kept
                        -x.get('timestamp', 0)  # Newer items kept
                    ))

                    # Remove oldest items until we're under the limit
                    size_removed = 0
                    items_removed = 0
                    new_items = []

                    for item in self.metadata['items']:
                        item_size = item.get('size', 0) / (1024 * 1024)  # MB

                        if queue_size - size_removed <= max_size_mb:
                            # We're under the limit, keep the rest
                            new_items.append(item)
                        else:
                            # Remove this item
                            size_removed += item_size
                            items_removed += 1

                    self.metadata['items'] = new_items
                    removed_count += items_removed
                    self.logger.info(f"Removed {items_removed} items to stay under size limit")

            if removed_count > 0:
                self._save_metadata()
                self.logger.info(f"Removed {removed_count} old items from queue")

                # Cleanup orphaned data files
                self._cleanup_data_files()

            return removed_count

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0

    def _calculate_queue_size(self):
        """
        Calculate total queue size in bytes

        Returns:
            Total size in bytes
        """
        total_size = 0

        # Add metadata file size
        if self.metadata_file.exists():
            total_size += self.metadata_file.stat().st_size

        # Add data files size
        for file in self.data_dir.glob("*"):
            total_size += file.stat().st_size

        return total_size

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
        # Count items by priority
        priorities = {}
        for item in self.metadata['items']:
            priority = item.get('priority', 100)
            if priority not in priorities:
                priorities[priority] = 0
            priorities[priority] += 1

        # Count by upload status
        total_items = len(self.metadata['items'])
        uploaded_items = sum(1 for item in self.metadata['items'] if item.get('uploaded', False))
        pending_items = total_items - uploaded_items

        # Count retry statistics
        retry_counts = {}
        for item in self.metadata['items']:
            if not item.get('uploaded', False):
                retry_count = item.get('retry_count', 0)
                if retry_count not in retry_counts:
                    retry_counts[retry_count] = 0
                retry_counts[retry_count] += 1

        # Calculate queue size
        queue_size = self._calculate_queue_size()

        # Get upload progress if in progress
        upload_progress = None
        if self.is_uploading:
            upload_progress = self.upload_progress

        return {
            "total_items": total_items,
            "uploaded_items": uploaded_items,
            "pending_items": pending_items,
            "priorities": priorities,
            "retry_counts": retry_counts,
            "queue_size_bytes": queue_size,
            "queue_size_mb": queue_size / (1024 * 1024),
            "in_memory_items": self.queue.qsize(),
            "upload_in_progress": self.is_uploading,
            "upload_progress": upload_progress,
            "compression_enabled": self.metadata.get('compression', True),
            "progressive_jpeg_enabled": self.metadata.get('progressive_jpeg', True)
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
                priority, item = self.queue.get(block=False)
                self._save_item_to_disk(item, priority)
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
        self.logger.info("Shutting down enhanced offline queue")

        # Cancel any ongoing upload
        if self.is_uploading:
            self.cancel_upload()

        # Stop background thread
        self.stop_event.set()
        if self.dump_thread.is_alive():
            self.dump_thread.join(timeout=5.0)

        # Flush remaining items
        flushed = self.flush()
        self.logger.info(f"Flushed {flushed} items to disk during shutdown")

        # Final save of metadata
        self._save_metadata()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("QueueTest")

    # Create queue
    queue = EnhancedOfflineQueue("enhanced_queue")

    # Add some test items
    for i in range(5):
        # Create a test image (black canvas with frame number)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Random priority
        priority = random.choice([10, 50, 100])  # Critical, High, Normal

        # Add to queue
        queue.add(
            frame_id=i,
            detections=[{"class": "person", "confidence": 0.95, "bbox": [10, 10, 100, 200]}],
            image=img,
            metadata={"source": "test"},
            priority=priority
        )
        logger.info(f"Added frame {i} with priority {priority}")

    # Show stats
    stats = queue.stats()
    logger.info(f"Queue stats: {stats}")

    # Clean up
    queue.shutdown()