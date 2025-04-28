#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Augmenter for Smart Object Tracking System.
Enhances small datasets with advanced augmentation techniques.
"""

import os
import sys
import shutil
import random
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


class DatasetAugmenter:
    """Augments object detection datasets to improve model training with small data"""

    def __init__(self, dataset_dir):
        """
        Initialize dataset augmenter.

        Args:
            dataset_dir: Path to dataset directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.logger = logging.getLogger('DatasetAugmenter')

        # Ensure dataset structure exists
        self.images_dir = self.dataset_dir / 'images'
        self.labels_dir = self.dataset_dir / 'labels'

        for split in ['train', 'val']:
            (self.images_dir / split).mkdir(parents=True, exist_ok=True)
            (self.labels_dir / split).mkdir(parents=True, exist_ok=True)

    def analyze_dataset(self):
        """
        Analyze dataset size and class distribution.

        Returns:
            dict: Dataset statistics
        """
        stats = {}

        # Count images in each split
        for split in ['train', 'val']:
            images_split_dir = self.images_dir / split
            labels_split_dir = self.labels_dir / split

            if images_split_dir.exists():
                image_files = list(images_split_dir.glob('*.jpg')) + list(images_split_dir.glob('*.png'))
                stats[f'{split}_images'] = len(image_files)
            else:
                stats[f'{split}_images'] = 0

            # Count annotations and class distribution
            class_counts = {}
            annotation_count = 0

            if labels_split_dir.exists():
                for label_file in labels_split_dir.glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    if class_id not in class_counts:
                                        class_counts[class_id] = 0
                                    class_counts[class_id] += 1
                                    annotation_count += 1
                    except Exception as e:
                        self.logger.warning(f"Error reading label file {label_file}: {e}")

            stats[f'{split}_annotations'] = annotation_count
            stats[f'{split}_class_distribution'] = class_counts

        return stats

    def augment_dataset(self, target_count=100, techniques=None, output_dir=None):
        """
        Augment dataset to reach target count per class.

        Args:
            target_count: Target number of samples per class
            techniques: List of augmentation techniques to use, or None for all
            output_dir: Output directory, or None to augment in place

        Returns:
            dict: Augmentation statistics
        """
        # Set default augmentation techniques if not specified
        if techniques is None:
            techniques = [
                'flip_horizontal', 'flip_vertical', 'rotate', 'brightness',
                'contrast', 'noise', 'blur'
            ]

        # Analyze current dataset
        stats = self.analyze_dataset()

        # Determine output directory
        if output_dir is None:
            # Augment in place
            output_images_dir = self.images_dir
            output_labels_dir = self.labels_dir
        else:
            # Create new dataset
            output_dir = Path(output_dir)
            output_images_dir = output_dir / 'images'
            output_labels_dir = output_dir / 'labels'

            # Create directory structure
            for split in ['train', 'val']:
                (output_images_dir / split).mkdir(parents=True, exist_ok=True)
                (output_labels_dir / split).mkdir(parents=True, exist_ok=True)

            # Copy original dataset
            for split in ['train', 'val']:
                # Copy images
                for img_file in (self.images_dir / split).glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy(img_file, output_images_dir / split)

                # Copy labels
                for label_file in (self.labels_dir / split).glob('*.txt'):
                    shutil.copy(label_file, output_labels_dir / split)

        # Get class distribution
        class_distribution = stats.get('train_class_distribution', {})

        # Focus on training set for augmentation
        augmentation_stats = {'original': stats, 'added': {}}

        # Process each class
        for class_id, count in class_distribution.items():
            if count >= target_count:
                self.logger.info(f"Class {class_id} already has {count} samples (target: {target_count})")
                continue

            # Calculate how many augmentations we need for this class
            needed = target_count - count
            self.logger.info(f"Class {class_id} needs {needed} more samples (current: {count})")

            # Get images containing this class
            images_with_class = self._get_images_with_class(class_id)

            if not images_with_class:
                self.logger.warning(f"No images found containing class {class_id}")
                continue

            # Generate augmentations
            added = self._generate_augmentations(
                images_with_class,
                needed,
                techniques,
                output_images_dir,
                output_labels_dir,
                class_id
            )

            augmentation_stats['added'][class_id] = added

        # Calculate final statistics
        augmentation_stats['final'] = self.analyze_dataset()

        return augmentation_stats

    def _get_images_with_class(self, class_id):
        """
        Get list of images containing a specific class.

        Args:
            class_id: Class ID to search for

        Returns:
            list: List of (image_path, label_path) tuples
        """
        images_with_class = []

        # Search in the training set
        train_img_dir = self.images_dir / 'train'
        train_lbl_dir = self.labels_dir / 'train'

        for label_file in train_lbl_dir.glob('*.txt'):
            base_name = label_file.stem

            # Check if this label file contains the class
            has_class = False
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and int(parts[0]) == class_id:
                            has_class = True
                            break
            except Exception:
                continue

            if has_class:
                # Find corresponding image
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = train_img_dir / f"{base_name}{ext}"
                    if img_path.exists():
                        images_with_class.append((img_path, label_file))
                        break

        return images_with_class

    def _generate_augmentations(self, source_images, count, techniques, output_images_dir, output_labels_dir,
                                target_class_id=None):
        """
        Generate augmented images.

        Args:
            source_images: List of (image_path, label_path) tuples
            count: Number of augmentations to generate
            techniques: List of augmentation techniques to use
            output_images_dir: Output directory for images
            output_labels_dir: Output directory for labels
            target_class_id: Class ID to focus on, or None for all classes

        Returns:
            int: Number of augmentations generated
        """
        if not source_images:
            return 0

        generated = 0
        pbar = tqdm(total=count, desc="Generating augmentations")

        while generated < count:
            # Select a random source image
            img_path, label_path = random.choice(source_images)

            # Load the image and labels
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                height, width = img.shape[:2]

                with open(label_path, 'r') as f:
                    label_lines = f.read().strip().split('\n')
            except Exception as e:
                self.logger.warning(f"Error loading {img_path}: {e}")
                continue

            # Parse labels
            boxes = []
            for line in label_lines:
                try:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    # Convert to absolute coordinates
                    x1 = int((x_center - box_width / 2) * width)
                    y1 = int((y_center - box_height / 2) * height)
                    x2 = int((x_center + box_width / 2) * width)
                    y2 = int((y_center + box_height / 2) * height)

                    boxes.append((class_id, x1, y1, x2, y2))
                except Exception:
                    continue

            # Skip if no valid boxes or target class not found
            if not boxes:
                continue

            if target_class_id is not None and not any(box[0] == target_class_id for box in boxes):
                continue

            # Apply random augmentations
            aug_img, aug_boxes = self._apply_augmentations(img, boxes, techniques)

            if aug_img is None or not aug_boxes:
                continue

            # Generate output file names
            timestamp = int(random.random() * 1000000)
            base_name = f"{img_path.stem}_aug_{timestamp}"
            aug_img_path = output_images_dir / 'train' / f"{base_name}.jpg"
            aug_label_path = output_labels_dir / 'train' / f"{base_name}.txt"

            # Save the augmented image
            try:
                cv2.imwrite(str(aug_img_path), aug_img)

                # Convert boxes back to YOLO format and save labels
                aug_height, aug_width = aug_img.shape[:2]
                with open(aug_label_path, 'w') as f:
                    for class_id, x1, y1, x2, y2 in aug_boxes:
                        # Convert to YOLO format
                        x_center = (x1 + x2) / (2 * aug_width)
                        y_center = (y1 + y2) / (2 * aug_height)
                        box_width = (x2 - x1) / aug_width
                        box_height = (y2 - y1) / aug_height

                        # Write label
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                generated += 1
                pbar.update(1)
            except Exception as e:
                self.logger.warning(f"Error saving augmentation: {e}")
                continue

            # Break if we've generated enough
            if generated >= count:
                break

        pbar.close()
        return generated

    def _apply_augmentations(self, img, boxes, techniques):
        """
        Apply random augmentations to an image and its bounding boxes.

        Args:
            img: Image array (H, W, C)
            boxes: List of (class_id, x1, y1, x2, y2) tuples
            techniques: List of allowed augmentation techniques

        Returns:
            tuple: (augmented_image, augmented_boxes)
        """
        height, width = img.shape[:2]
        aug_img = img.copy()
        aug_boxes = boxes.copy()

        # Select 1-3 random augmentation techniques
        num_augs = random.randint(1, min(3, len(techniques)))
        selected_techniques = random.sample(techniques, num_augs)

        # Apply selected augmentations
        for technique in selected_techniques:
            if technique == 'flip_horizontal' and 'flip_horizontal' in techniques:
                # Horizontal flip
                aug_img = cv2.flip(aug_img, 1)  # 1 for horizontal flip

                # Update boxes
                new_boxes = []
                for class_id, x1, y1, x2, y2 in aug_boxes:
                    new_x1 = width - x2
                    new_x2 = width - x1
                    new_boxes.append((class_id, new_x1, y1, new_x2, y2))
                aug_boxes = new_boxes

            elif technique == 'flip_vertical' and 'flip_vertical' in techniques:
                # Vertical flip
                aug_img = cv2.flip(aug_img, 0)  # 0 for vertical flip

                # Update boxes
                new_boxes = []
                for class_id, x1, y1, x2, y2 in aug_boxes:
                    new_y1 = height - y2
                    new_y2 = height - y1
                    new_boxes.append((class_id, x1, new_y1, x2, new_y2))
                aug_boxes = new_boxes

            elif technique == 'rotate' and 'rotate' in techniques:
                # Random rotation between -15 and 15 degrees
                angle = random.uniform(-15, 15)
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                aug_img = cv2.warpAffine(aug_img, rotation_matrix, (width, height),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

                # Update boxes (simplified rotation of corners and rebounding)
                new_boxes = []
                for class_id, x1, y1, x2, y2 in aug_boxes:
                    # Get corners
                    corners = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32)

                    # Adjust for rotation center
                    corners -= [center[0], center[1]]

                    # Rotate corners
                    theta = np.radians(angle)
                    rot_mat = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]
                    ])

                    corners = corners @ rot_mat.T

                    # Readjust for rotation center
                    corners += [center[0], center[1]]

                    # Get bounds
                    new_x1 = max(0, int(np.min(corners[:, 0])))
                    new_y1 = max(0, int(np.min(corners[:, 1])))
                    new_x2 = min(width, int(np.max(corners[:, 0])))
                    new_y2 = min(height, int(np.max(corners[:, 1])))

                    # Add to boxes if valid
                    if new_x2 > new_x1 and new_y2 > new_y1:
                        new_boxes.append((class_id, new_x1, new_y1, new_x2, new_y2))

                aug_boxes = new_boxes

            elif technique == 'brightness' and 'brightness' in techniques:
                # Random brightness adjustment
                factor = random.uniform(0.5, 1.5)
                hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV)
                hsv = hsv.astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
                hsv = hsv.astype(np.uint8)
                aug_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            elif technique == 'contrast' and 'contrast' in techniques:
                # Random contrast adjustment
                factor = random.uniform(0.5, 1.5)
                mean = np.mean(aug_img, axis=(0, 1), keepdims=True)
                aug_img = np.clip((aug_img - mean) * factor + mean, 0, 255).astype(np.uint8)

            elif technique == 'noise' and 'noise' in techniques:
                # Add random noise
                noise = np.random.normal(0, random.uniform(5, 20), aug_img.shape).astype(np.float32)
                aug_img = np.clip(aug_img + noise, 0, 255).astype(np.uint8)

            elif technique == 'blur' and 'blur' in techniques:
                # Random blur
                kernel_size = random.choice([3, 5])
                aug_img = cv2.GaussianBlur(aug_img, (kernel_size, kernel_size), 0)

        # Validate bounding boxes
        valid_boxes = []
        for class_id, x1, y1, x2, y2 in aug_boxes:
            # Ensure coordinates are within image bounds
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))

            # Ensure minimum size
            if x2 - x1 >= 5 and y2 - y1 >= 5:
                valid_boxes.append((class_id, x1, y1, x2, y2))

        if not valid_boxes:
            # If no valid boxes, return None
            return None, None

        return aug_img, valid_boxes

    def create_mosaic_image(self, source_images, output_dir, grid_size=2):
        """
        Create mosaic images by combining multiple images.

        Args:
            source_images: List of (image_path, label_path) tuples
            output_dir: Output directory for images and labels
            grid_size: Size of mosaic grid (2 for 2x2, 3 for 3x3)

        Returns:
            tuple: Path to created image and label file
        """
        if len(source_images) < grid_size * grid_size:
            return None, None

        # Select random images
        selected_images = random.sample(source_images, grid_size * grid_size)

        # Determine target size
        target_size = 640  # Standard size
        mosaic_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Cell size
        cell_size = target_size // grid_size

        all_boxes = []

        # Place images in grid
        for i, (img_path, label_path) in enumerate(selected_images):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Resize to cell size
                img_resized = cv2.resize(img, (cell_size, cell_size))

                # Calculate grid position
                grid_x = i % grid_size
                grid_y = i // grid_size

                # Place in mosaic
                mosaic_img[
                grid_y * cell_size:(grid_y + 1) * cell_size,
                grid_x * cell_size:(grid_x + 1) * cell_size
                ] = img_resized

                # Read and transform labels
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])

                        # Transform to mosaic coordinates
                        new_x_center = (grid_x + x_center) / grid_size
                        new_y_center = (grid_y + y_center) / grid_size
                        new_width = box_width / grid_size
                        new_height = box_height / grid_size

                        all_boxes.append((class_id, new_x_center, new_y_center, new_width, new_height))

            except Exception as e:
                self.logger.warning(f"Error processing {img_path} for mosaic: {e}")
                continue

        if not all_boxes:
            return None, None

        # Generate output paths
        timestamp = int(random.random() * 1000000)
        base_name = f"mosaic_{timestamp}"
        mosaic_img_path = output_dir / 'train' / f"{base_name}.jpg"
        mosaic_label_path = output_dir / 'train' / f"{base_name}.txt"

        # Save mosaic image and labels
        try:
            cv2.imwrite(str(mosaic_img_path), mosaic_img)

            with open(mosaic_label_path, 'w') as f:
                for class_id, x_center, y_center, width, height in all_boxes:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            return mosaic_img_path, mosaic_label_path

        except Exception as e:
            self.logger.warning(f"Error saving mosaic: {e}")
            return None, None

    def generate_mosaics(self, count=10, grid_size=2):
        """
        Generate multiple mosaic images.

        Args:
            count: Number of mosaic images to generate
            grid_size: Size of mosaic grid

        Returns:
            int: Number of generated mosaics
        """
        # Get all training images
        train_img_dir = self.images_dir / 'train'
        train_lbl_dir = self.labels_dir / 'train'

        source_images = []
        for label_file in train_lbl_dir.glob('*.txt'):
            base_name = label_file.stem

            # Find corresponding image
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = train_img_dir / f"{base_name}{ext}"
                if img_path.exists():
                    source_images.append((img_path, label_file))
                    break

        if len(source_images) < grid_size * grid_size:
            self.logger.warning(f"Not enough images for creating mosaics (need at least {grid_size * grid_size})")
            return 0

        # Generate mosaics
        generated = 0
        for _ in tqdm(range(count), desc="Generating mosaics"):
            _, _ = self.create_mosaic_image(source_images, self.images_dir, grid_size)
            generated += 1

        return generated


if __name__ == "__main__":
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Dataset Augmentation for Small Datasets")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--output', type=str, help="Output directory (if different from dataset)")
    parser.add_argument('--target', type=int, default=100, help="Target number of samples per class")
    parser.add_argument('--mosaics', type=int, default=10, help="Number of mosaic images to generate")
    parser.add_argument('--analyze-only', action='store_true', help="Only analyze dataset without augmentation")
    parser.add_argument('--techniques', type=str,
                        default="flip_horizontal,flip_vertical,rotate,brightness,contrast,noise,blur",
                        help="Comma-separated list of augmentation techniques to use")

    args = parser.parse_args()

    # Create augmenter
    augmenter = DatasetAugmenter(args.dataset)

    # Analyze dataset
    stats = augmenter.analyze_dataset()

    print("\nDataset Statistics:")
    print(f"Training images: {stats.get('train_images', 0)}")
    print(f"Validation images: {stats.get('val_images', 0)}")
    print(f"Training annotations: {stats.get('train_annotations', 0)}")
    print(f"Validation annotations: {stats.get('val_annotations', 0)}")

    print("\nClass distribution (training):")
    for class_id, count in stats.get('train_class_distribution', {}).items():
        print(f"  Class {class_id}: {count} annotations")

    if args.analyze_only:
        print("\nAnalysis complete. Exiting without augmentation.")
        sys.exit(0)

    # Parse techniques
    techniques = args.techniques.split(',')

    # Perform augmentation
    print(f"\nAugmenting dataset to {args.target} samples per class...")
    aug_stats = augmenter.augment_dataset(
        target_count=args.target,
        techniques=techniques,
        output_dir=args.output
    )

    # Generate mosaics
    if args.mosaics > 0:
        print(f"\nGenerating {args.mosaics} mosaic images...")
        mosaic_count = augmenter.generate_mosaics(count=args.mosaics)
        print(f"Generated {mosaic_count} mosaic images")

    # Print final statistics
    final_stats = aug_stats.get('final', {})
    print("\nFinal Dataset Statistics:")
    print(f"Training images: {final_stats.get('train_images', 0)}")
    print(f"Validation images: {final_stats.get('val_images', 0)}")
    print(f"Training annotations: {final_stats.get('train_annotations', 0)}")
    print(f"Validation annotations: {final_stats.get('val_annotations', 0)}")

    print("\nFinal class distribution (training):")
    for class_id, count in final_stats.get('train_class_distribution', {}).items():
        print(f"  Class {class_id}: {count} annotations")

    print("\nAugmentation complete!")