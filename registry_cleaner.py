# registry_cleaner.py

import os
import shutil
from pathlib import Path

# Configuration
REGISTRY_DIR = Path("models/registry")

# Main cleaner logic
def clean_registry(dry_run=True):
    if not REGISTRY_DIR.exists():
        print(f"Registry folder not found: {REGISTRY_DIR}")
        return

    broken_models = []

    for model_dir in REGISTRY_DIR.glob("model_*"):
        if model_dir.is_dir():
            summary_file = model_dir / "summary.json"
            onnx_file = model_dir / "model.onnx"
            training_curve = model_dir / "training_curves.png"

            missing = []
            if not summary_file.exists():
                missing.append("summary.json")
            if not onnx_file.exists():
                missing.append("model.onnx")
            if not training_curve.exists():
                missing.append("training_curves.png")

            if missing:
                broken_models.append((model_dir, missing))

    if not broken_models:
        print("\n✅ No broken models found. Registry is clean.")
        return

    print("\n⚠️ Broken models detected:")
    for model_dir, missing_files in broken_models:
        print(f"- {model_dir.name} missing: {', '.join(missing_files)}")

    if dry_run:
        print("\nDry run mode: No folders deleted.")
    else:
        print("\nDeleting broken models...")
        for model_dir, _ in broken_models:
            shutil.rmtree(model_dir)
            print(f"Deleted {model_dir.name}")

        print("\n🧹 Cleaning completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean broken models from registry.")
    parser.add_argument("--delete", action="store_true", help="Actually delete broken models (default is dry run)")
    args = parser.parse_args()

    clean_registry(dry_run=not args.delete)
