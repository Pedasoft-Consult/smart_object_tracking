#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Manager for Smart Object Tracking System.
Handles dataset creation, annotation, and management for model training.
"""

import os
import json
import time
import random
import shutil
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import yaml


class DatasetManager:
    """
    Manages datasets for object detection model training.
    Supports YOLO format and provides utilities for dataset preparation.
    """

    def __init__(self, dataset_dir, annotation_format='yolo', class_file=None):
        """
        Initialize the dataset manager.

        Args:
            dataset_dir: Directory to store dataset
            annotation_format: Format of annotations ('yolo', 'coco', etc.)
            class_file: Path to file containing class names (one per line)
        """
        self.dataset_dir = Path(dataset_dir)
        self.annotation_format = annotation_format
        self.metadata_file = self.dataset_dir / "dataset_metadata.json"
        self.logger = logging.getLogger('DatasetManager')

        # Initialize class list
        self.classes = []
        if class_file and os.path.exists(class_file):
            with open(class_file, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]

        # Initialize directory structure and metadata
        self._initialize_directories()
        self._load_metadata()

        self.logger.info(f"Dataset manager initialized at {self.dataset_dir}")
        self.logger.info(f"Using {len(self.classes)} classes: {self.classes}")

    def _initialize_directories(self):
        """Create dataset directory structure"""
        # Create main directories
        (self.dataset_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / 'labels' / 'test').mkdir(parents=True, exist_ok=True)

        # Create additional directories for organization
        (self.dataset_dir / 'feedback').mkdir(exist_ok=True)  # For feedback images
        (self.dataset_dir / 'exports').mkdir(exist_ok=True)  # For exported datasets

    def _load_metadata(self):
        """Load dataset metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    # Update class list from metadata if available
                    if 'classes' in self.metadata and self.metadata['classes']:
                        self.classes = self.metadata['classes']
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}, creating new file")
                self._create_default_metadata()
        else:
            self._create_default_metadata()

    def _create_default_metadata(self):
        """Create default metadata structure"""
        self.metadata = {
            "created": time.time(),
            "last_updated": time.time(),
            "version": 1,
            "classes": self.classes,
            "stats": {
                "train": {"images": 0, "annotations": 0},
                "val": {"images": 0, "annotations": 0},
                "test": {"images": 0, "annotations": 0}
            },
            "splits": {
                "train": [],
                "val": [],
                "test": []
            },
            "history": []
        }
        self._save_metadata()

    def _save_metadata(self):
        """Save dataset metadata"""
        try:
            self.metadata["last_updated"] = time.time()
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")

    def _update_class_names(self):
        """Update class names file in dataset"""
        with open(self.dataset_dir / 'classes.txt', 'w') as f:
            for class_name in self.classes:
                f.write(f"{class_name}\n")

        # Also update metadata
        self.metadata["classes"] = self.classes
        self._save_metadata()

    def add_class(self, class_name):
        """
        Add a new class to the dataset.

        Args:
            class_name: Name of the class to add

        Returns:
            int: Index of the class
        """
        if class_name not in self.classes:
            self.classes.append(class_name)
            self._update_class_names()
            self.logger.info(f"Added new class: {class_name} (index {len(self.classes) - 1})")
            return len(self.classes) - 1
        else:
            return self.classes.index(class_name)

    def _get_class_id(self, class_name):
        """
        Get class ID from name.

        Args:
            class_name: Class name to look up

        Returns:
            int: Class ID or -1 if not found
        """
        if class_name in self.classes:
            return self.classes.index(class_name)
        return -1

    def _convert_to_yolo_format(self, image_width, image_height, bbox, class_id):
        """
        Convert [x1, y1, x2, y2] bbox to YOLO format [class_id, center_x, center_y, width, height].

        Args:
            image_width: Width of the image
            image_height: Height of the image
            bbox: Bounding box in [x1, y1, x2, y2] format
            class_id: Class ID

        Returns:
            list: Annotation in YOLO format [class_id, center_x, center_y, width, height]
        """
        x1, y1, x2, y2 = bbox
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, image_width))
        y1 = max(0, min(y1, image_height))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        # Calculate center and dimensions normalized to image size
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        # Return YOLO format annotation
        return [class_id, center_x, center_y, width, height]

    def _convert_from_yolo_format(self, image_width, image_height, yolo_annotation):
        """
        Convert YOLO format [class_id, center_x, center_y, width, height] to [x1, y1, x2, y2, class_id].

        Args:
            image_width: Width of the image
            image_height: Height of the image
            yolo_annotation: Annotation in YOLO format [class_id, center_x, center_y, width, height]

        Returns:
            list: Bounding box in [x1, y1, x2, y2, class_id] format
        """
        class_id, center_x, center_y, width, height = yolo_annotation

        # Convert normalized coordinates to pixel values
        center_x *= image_width
        center_y *= image_height
        width *= image_width
        height *= image_height

        # Calculate bounding box coordinates
        x1 = center_x - (width / 2)
        y1 = center_y - (height / 2)
        x2 = center_x + (width / 2)
        y2 = center_y + (height / 2)

        return [x1, y1, x2, y2, class_id]

    def add_from_feedback(self, image, annotations, split='train', image_id=None):
        """
        Add image and annotations from feedback system.

        Args:
            image: Image as numpy array
            annotations: List of annotations in format [x1, y1, x2, y2, class_id] or
                        [{"bbox": [x1, y1, x2, y2], "class_id": id}]
            split: Dataset split ('train', 'val', 'test')
            image_id: Unique identifier for the image (generated if None)

        Returns:
            str: Image ID
        """
        try:
            # Validate split
            if split not in ['train', 'val', 'test']:
                split = 'train'  # Default to train

            # Generate image ID if not provided
            if image_id is None:
                image_id = f"feedback_{int(time.time())}_{random.randint(1000, 9999)}"

            # Define paths
            img_path = self.dataset_dir / 'images' / split / f"{image_id}.jpg"
            label_path = self.dataset_dir / 'labels' / split / f"{image_id}.txt"

            # Save image
            cv2.imwrite(str(img_path), image)

            # Get image dimensions
            height, width = image.shape[:2]

            # Process annotations
            yolo_annotations = []
            for ann in annotations:
                if isinstance(ann, dict):
                    # Handle dictionary format
                    bbox = ann.get('bbox', [0, 0, 0, 0])
                    class_id = ann.get('class_id', 0)
                    if 'class_name' in ann and ann['class_name'] not in self.classes:
                        class_name = ann['class_name']
                        class_id = self.add_class(class_name)
                elif isinstance(ann, (list, tuple)) and len(ann) >= 5:
                    # Handle array format [x1, y1, x2, y2, class_id]
                    bbox = ann[:4]
                    class_id = ann[4]
                else:
                    continue  # Skip invalid annotations

                # Convert to YOLO format
                yolo_ann = self._convert_to_yolo_format(width, height, bbox, class_id)
                yolo_annotations.append(yolo_ann)

            # Write annotations to file
            with open(label_path, 'w') as f:
                for yolo_ann in yolo_annotations:
                    f.write(
                        f"{int(yolo_ann[0])} {yolo_ann[1]:.6f} {yolo_ann[2]:.6f} {yolo_ann[3]:.6f} {yolo_ann[4]:.6f}\n")

            # Update metadata
            self.metadata["splits"][split].append(image_id)
            self.metadata["stats"][split]["images"] += 1
            self.metadata["stats"][split]["annotations"] += len(yolo_annotations)

            # Add to history
            self.metadata["history"].append({
                "timestamp": time.time(),
                "action": "add_image",
                "image_id": image_id,
                "split": split,
                "annotation_count": len(yolo_annotations)
            })

            self._save_metadata()

            self.logger.info(f"Added image {image_id} to {split} split with {len(yolo_annotations)} annotations")
            return image_id

        except Exception as e:
            self.logger.error(f"Error adding feedback image: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def add_from_file(self, image_path, annotation_path=None, split='train', image_id=None):
        """
        Add image and annotations from files.

        Args:
            image_path: Path to image file
            annotation_path: Path to annotation file (in supported format)
            split: Dataset split ('train', 'val', 'test')
            image_id: Unique identifier for the image (generated if None)

        Returns:
            str: Image ID
        """
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to read image: {image_path}")
                return None

            # Generate image ID if not provided
            if image_id is None:
                image_id = f"file_{int(time.time())}_{random.randint(1000, 9999)}"

            # Read annotations if provided
            annotations = []
            if annotation_path and os.path.exists(annotation_path):
                # Determine annotation format
                if annotation_path.endswith('.txt'):  # YOLO format
                    height, width = image.shape[:2]
                    with open(annotation_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                center_x = float(parts[1])
                                center_y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])

                                # Convert to [x1, y1, x2, y2, class_id] format
                                x1 = (center_x - w / 2) * width
                                y1 = (center_y - h / 2) * height
                                x2 = (center_x + w / 2) * width
                                y2 = (center_y + h / 2) * height

                                annotations.append([x1, y1, x2, y2, class_id])
                elif annotation_path.endswith('.json'):  # JSON format
                    with open(annotation_path, 'r') as f:
                        ann_data = json.load(f)
                        # Handle different JSON annotation formats
                        if isinstance(ann_data, dict) and 'annotations' in ann_data:
                            # COCO-like format
                            for ann in ann_data['annotations']:
                                if 'bbox' in ann and 'category_id' in ann:
                                    x, y, w, h = ann['bbox']
                                    class_id = ann['category_id']
                                    annotations.append([x, y, x + w, y + h, class_id])
                        elif isinstance(ann_data, list):
                            # Simple list of annotations
                            for ann in ann_data:
                                if 'bbox' in ann and 'class_id' in ann:
                                    bbox = ann['bbox']
                                    class_id = ann['class_id']
                                    annotations.append([bbox[0], bbox[1], bbox[2], bbox[3], class_id])

            # Add to dataset
            return self.add_from_feedback(image, annotations, split, image_id)

        except Exception as e:
            self.logger.error(f"Error adding file: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def add_from_directory(self, image_dir, annotation_dir=None, split='train', recursive=False):
        """
        Add all images and annotations from directories.

        Args:
            image_dir: Directory containing images
            annotation_dir: Directory containing annotations (same filenames but different extension)
            split: Dataset split ('train', 'val', 'test')
            recursive: Whether to search directories recursively

        Returns:
            int: Number of images added
        """
        count = 0
        try:
            # Get all image files
            image_files = []
            if recursive:
                for root, _, files in os.walk(image_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            image_files.append(os.path.join(root, file))
            else:
                image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

            # Process each image
            for img_path in image_files:
                # Find corresponding annotation file
                ann_path = None
                if annotation_dir:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    # Try different annotation extensions
                    for ext in ['.txt', '.json', '.xml']:
                        candidate = os.path.join(annotation_dir, base_name + ext)
                        if os.path.exists(candidate):
                            ann_path = candidate
                            break

                # Add to dataset
                image_id = self.add_from_file(img_path, ann_path, split)
                if image_id:
                    count += 1

            self.logger.info(f"Added {count} images from directory {image_dir}")
            return count

        except Exception as e:
            self.logger.error(f"Error adding from directory: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return count

    def get_image(self, image_id, split=None):
        """
        Get an image by ID.

        Args:
            image_id: Image ID
            split: Dataset split to look in (searches all if None)

        Returns:
            tuple: (image, annotations, split) or (None, None, None) if not found
        """
        # If split is specified, only look there
        if split:
            splits = [split]
        else:
            splits = ['train', 'val', 'test']

        for s in splits:
            img_path = self.dataset_dir / 'images' / s / f"{image_id}.jpg"
            if img_path.exists():
                # Load image
                image = cv2.imread(str(img_path))

                # Load annotations
                annotations = []
                label_path = self.dataset_dir / 'labels' / s / f"{image_id}.txt"
                if label_path.exists():
                    height, width = image.shape[:2]
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                yolo_ann = [int(parts[0]), float(parts[1]), float(parts[2]),
                                            float(parts[3]), float(parts[4])]
                                bbox_ann = self._convert_from_yolo_format(width, height, yolo_ann)
                                annotations.append(bbox_ann)

                return image, annotations, s

        return None, None, None

    def split_dataset(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, shuffle=True):
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Ratio of images to use for training
            val_ratio: Ratio of images to use for validation
            test_ratio: Ratio of images to use for testing
            shuffle: Whether to shuffle images before splitting

        Returns:
            dict: Split statistics
        """
        try:
            # Normalize ratios
            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

            # Get all image IDs currently in the dataset
            all_images = []
            for split in ['train', 'val', 'test']:
                image_dir = self.dataset_dir / 'images' / split
                for img_file in image_dir.glob('*.jpg'):
                    img_id = img_file.stem
                    all_images.append((img_id, split))

            # Shuffle if requested
            if shuffle:
                random.shuffle(all_images)

            # Calculate split counts
            total_images = len(all_images)
            train_count = int(total_images * train_ratio)
            val_count = int(total_images * val_ratio)
            test_count = total_images - train_count - val_count

            # Reset split assignments
            self.metadata["splits"] = {
                "train": [],
                "val": [],
                "test": []
            }

            # Redistribute images
            for i, (img_id, current_split) in enumerate(all_images):
                if i < train_count:
                    new_split = 'train'
                elif i < train_count + val_count:
                    new_split = 'val'
                else:
                    new_split = 'test'

                # Move image and label files if the split has changed
                if new_split != current_split:
                    # Move image file
                    src_img = self.dataset_dir / 'images' / current_split / f"{img_id}.jpg"
                    dst_img = self.dataset_dir / 'images' / new_split / f"{img_id}.jpg"
                    if src_img.exists():
                        shutil.move(str(src_img), str(dst_img))

                    # Move label file
                    src_label = self.dataset_dir / 'labels' / current_split / f"{img_id}.txt"
                    dst_label = self.dataset_dir / 'labels' / new_split / f"{img_id}.txt"
                    if src_label.exists():
                        shutil.move(str(src_label), str(dst_label))

                # Update metadata
                self.metadata["splits"][new_split].append(img_id)

            # Update statistics
            self._update_statistics()

            # Add to history
            self.metadata["history"].append({
                "timestamp": time.time(),
                "action": "split_dataset",
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": test_ratio,
                "train_count": train_count,
                "val_count": val_count,
                "test_count": test_count
            })

            self._save_metadata()

            self.logger.info(f"Dataset split: train={train_count}, val={val_count}, test={test_count}")

            return {
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "total": total_images
            }

        except Exception as e:
            self.logger.error(f"Error splitting dataset: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "train": 0,
                "val": 0,
                "test": 0,
                "total": 0,
                "error": str(e)
            }

    def _update_statistics(self):
        """Update dataset statistics"""
        try:
            # Reset statistics
            stats = {
                "train": {"images": 0, "annotations": 0},
                "val": {"images": 0, "annotations": 0},
                "test": {"images": 0, "annotations": 0}
            }

            # Count images and annotations
            for split in ['train', 'val', 'test']:
                image_dir = self.dataset_dir / 'images' / split
                label_dir = self.dataset_dir / 'labels' / split

                # Count images
                image_count = sum(1 for _ in image_dir.glob('*.jpg'))
                stats[split]["images"] = image_count

                # Count annotations
                annotation_count = 0
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        annotation_count += sum(1 for _ in f)

                stats[split]["annotations"] = annotation_count

            # Update metadata
            self.metadata["stats"] = stats
            self._save_metadata()

        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")

    def export_dataset_yaml(self, output_path=None):
        """
        Create YAML configuration file for training with YOLOv5.

        Args:
            output_path: Path to save YAML file (defaults to dataset directory)

        Returns:
            str: Path to created YAML file
        """
        try:
            # Default output path
            if output_path is None:
                output_path = self.dataset_dir / 'dataset.yaml'

            # Create configuration
            dataset_config = {
                'path': str(self.dataset_dir),
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': len(self.classes),
                'names': self.classes
            }

            # Write to file
            with open(output_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

            self.logger.info(f"Exported dataset YAML configuration to {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Error exporting dataset YAML: {e}")
            return None

    def export_dataset(self, output_dir=None, format='yolov5'):
        """
        Export dataset to a specific format.

        Args:
            output_dir: Directory to export to (defaults to exports directory)
            format: Export format ('yolov5', 'coco', etc.)

        Returns:
            str: Path to exported dataset
        """
        try:
            # Default output directory
            if output_dir is None:
                output_dir = self.dataset_dir / 'exports' / f"{format}_{int(time.time())}"
            else:
                output_dir = Path(output_dir)

            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)

            if format.lower() == 'yolov5':
                # For YOLOv5, create symlinks to existing structure
                for split in ['train', 'val', 'test']:
                    # Create directories
                    (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
                    (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

                    # Link or copy images and labels
                    for img_path in (self.dataset_dir / 'images' / split).glob('*.jpg'):
                        dst_path = output_dir / 'images' / split / img_path.name
                        if not dst_path.exists():
                            try:
                                os.symlink(img_path, dst_path)
                            except:
                                shutil.copy(img_path, dst_path)

                    for label_path in (self.dataset_dir / 'labels' / split).glob('*.txt'):
                        dst_path = output_dir / 'labels' / split / label_path.name
                        if not dst_path.exists():
                            try:
                                os.symlink(label_path, dst_path)
                            except:
                                shutil.copy(label_path, dst_path)

                # Create classes.txt file
                with open(output_dir / 'classes.txt', 'w') as f:
                    for class_name in self.classes:
                        f.write(f"{class_name}\n")

                # Create dataset YAML
                self.export_dataset_yaml(output_dir / 'dataset.yaml')

            elif format.lower() == 'coco':
                # For COCO format, convert annotations to COCO JSON format
                coco_data = {
                    "info": {
                        "year": time.strftime("%Y"),
                        "version": "1.0",
                        "description": "Exported from Smart Object Tracking System",
                        "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "licenses": [
                        {
                            "id": 1,
                            "name": "Unknown",
                            "url": ""
                        }
                    ],
                    "categories": [
                        {"id": i, "name": name, "supercategory": "object"}
                        for i, name in enumerate(self.classes)
                    ],
                    "images": [],
                    "annotations": []
                }

                # Structure for each split
                for split in ['train', 'val', 'test']:
                    split_dir = output_dir / split
                    split_dir.mkdir(parents=True, exist_ok=True)

                    # Reset annotation ID for each split
                    ann_id = 1
                    split_coco = coco_data.copy()
                    split_coco["images"] = []
                    split_coco["annotations"] = []

                    # Process all images in this split
                    img_id = 1
                    for img_path in (self.dataset_dir / 'images' / split).glob('*.jpg'):
                        # Copy image
                        shutil.copy(img_path, split_dir / img_path.name)

                        # Read image to get dimensions
                        image = Image.open(img_path)
                        width, height = image.size

                        # Add image info
                        image_info = {
                            "id": img_id,
                            "file_name": img_path.name,
                            "width": width,
                            "height": height
                        }
                        split_coco["images"].append(image_info)

                        # Get annotations
                        label_path = self.dataset_dir / 'labels' / split / f"{img_path.stem}.txt"
                        if label_path.exists():
                            with open(label_path, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) == 5:
                                        class_id = int(parts[0])
                                        center_x = float(parts[1])
                                        center_y = float(parts[2])
                                        w = float(parts[3])
                                        h = float(parts[4])

                                        # Convert to COCO format (absolute coordinates)
                                        x = (center_x - w / 2) * width
                                        y = (center_y - h / 2) * height
                                        w = w * width
                                        h = h * height

                                        # Add annotation
                                        annotation = {
                                            "id": ann_id,
                                            "image_id": img_id,
                                            "category_id": class_id,
                                            "bbox": [x, y, w, h],
                                            "area": w * h,
                                            "segmentation": [],
                                            "iscrowd": 0
                                        }
                                        split_coco["annotations"].append(annotation)
                                        ann_id += 1

                        img_id += 1

                    # Write COCO JSON for this split
                    with open(split_dir / f"{split}.json", 'w') as f:
                        json.dump(split_coco, f, indent=2)

            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None

            self.logger.info(f"Dataset exported to {output_dir} in {format} format")
            return str(output_dir)

        except Exception as e:
            self.logger.error(f"Error exporting dataset: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def cleanup(self):
        """
        Clean up dataset by removing invalid images and annotations.

        Returns:
            dict: Cleanup statistics
        """
        try:
            stats = {
                "removed_images": 0,
                "removed_labels": 0,
                "orphaned_labels": 0,
                "orphaned_images": 0
            }

            # Process each split
            for split in ['train', 'val', 'test']:
                # Check for orphaned labels (labels without images)
                image_files = set(p.stem for p in (self.dataset_dir / 'images' / split).glob('*.jpg'))
                label_files = set(p.stem for p in (self.dataset_dir / 'labels' / split).glob('*.txt'))

                # Find orphaned labels
                orphaned_labels = label_files - image_files
                for label_id in orphaned_labels:
                    label_path = self.dataset_dir / 'labels' / split / f"{label_id}.txt"
                    if label_path.exists():
                        label_path.unlink()
                        stats["orphaned_labels"] += 1

                # Find orphaned images
                orphaned_images = image_files - label_files
                for img_id in orphaned_images:
                    img_path = self.dataset_dir / 'images' / split / f"{img_id}.jpg"
                    if img_path.exists():
                        stats["orphaned_images"] += 1
                        # Don't delete orphaned images, they might be valid but unannotated

            # Update metadata
            self._update_statistics()

            # Add to history
            self.metadata["history"].append({
                "timestamp": time.time(),
                "action": "cleanup",
                "stats": stats
            })

            self._save_metadata()

            self.logger.info(f"Dataset cleanup completed: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Error during dataset cleanup: {e}")
            return {"error": str(e)}

    def get_statistics(self):
        """
        Get dataset statistics.

        Returns:
            dict: Dataset statistics
        """
        # Update statistics first
        self._update_statistics()

        # Gather class distribution
        class_distribution = {class_name: 0 for class_name in self.classes}

        for split in ['train', 'val', 'test']:
            label_dir = self.dataset_dir / 'labels' / split

            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 1:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(self.classes):
                                class_name = self.classes[class_id]
                                class_distribution[class_name] += 1

        # Compile complete statistics
        stats = {
            "dataset_dir": str(self.dataset_dir),
            "classes": self.classes,
            "class_count": len(self.classes),
            "class_distribution": class_distribution,
            "splits": self.metadata["stats"],
            "total_images": sum(s["images"] for s in self.metadata["stats"].values()),
            "total_annotations": sum(s["annotations"] for s in self.metadata["stats"].values()),
            "last_updated": self.metadata["last_updated"]
        }

        return stats


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Initialize dataset manager
    dataset_manager = DatasetManager("datasets/object_detection")

    # Add some test classes
    dataset_manager.add_class("person")
    dataset_manager.add_class("car")
    dataset_manager.add_class("bicycle")

    # Print statistics
    stats = dataset_manager.get_statistics()
    print(f"Dataset statistics: {stats}")