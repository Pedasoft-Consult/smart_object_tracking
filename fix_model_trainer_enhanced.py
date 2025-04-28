import os
import shutil
from pathlib import Path

class ModelTrainerEnhancedAutoFix:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)

    def ensure_validation_split(self):
        """
        Ensure that images/val and labels/val exist.
        If they don't exist, copy a few samples from train.
        """
        images_train = self.dataset_dir / 'images' / 'train'
        labels_train = self.dataset_dir / 'labels' / 'train'
        images_val = self.dataset_dir / 'images' / 'val'
        labels_val = self.dataset_dir / 'labels' / 'val'

        # Create val directories if they don't exist
        images_val.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

        # Check if val already has images
        val_images = list(images_val.glob('*.jpg')) + list(images_val.glob('*.png'))
        if len(val_images) == 0:
            # Copy a few train images and labels
            train_images = list(images_train.glob('*.jpg')) + list(images_train.glob('*.png'))
            train_labels = list(labels_train.glob('*.txt'))

            if len(train_images) == 0 or len(train_labels) == 0:
                print("[ERROR] No training images or labels found to create validation set!")
                return

            # Copy up to 5 samples
            sample_count = min(5, len(train_images))
            for img_path in train_images[:sample_count]:
                target_path = images_val / img_path.name
                shutil.copy(img_path, target_path)

            for lbl_path in train_labels[:sample_count]:
                target_path = labels_val / lbl_path.name
                shutil.copy(lbl_path, target_path)

            print(f"[INFO] Created validation set with {sample_count} samples.")
        else:
            print("[INFO] Validation set already exists. No action needed.")

if __name__ == "__main__":
    # Example usage
    dataset_dir = "/mnt/c/job_projects/smart_object_tracking/dataset"
    fixer = ModelTrainerEnhancedAutoFix(dataset_dir)
    fixer.ensure_validation_split()
