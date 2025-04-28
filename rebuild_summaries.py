#!/usr/bin/env python3
# rebuild_summaries.py

import os
import json
from pathlib import Path
import time

def rebuild_summaries(registry_dir):
    for model_dir in Path(registry_dir).glob("model_*"):
        summary_file = model_dir / "summary.json"
        model_file = model_dir / "model.pt"

        if not summary_file.exists():
            print(f"Creating summary for {model_dir.name}...")

            summary = {
                "training_id": "unknown",
                "model_id": model_dir.name,
                "model_type": "unknown",
                "epochs": "unknown",
                "batch_size": "unknown",
                "image_size": "unknown",
                "learning_rate": "unknown",
                "freeze_layers": "unknown",
                "early_stopping_patience": "unknown",
                "exported_model_path": str(model_dir / "model.onnx") if (model_dir / "model.onnx").exists() else None,
                "evaluation": {},
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"✅ Created {summary_file}")
        else:
            print(f"Summary already exists for {model_dir.name}")

def main():
    registry_dir = "models/registry"
    if not os.path.exists(registry_dir):
        print(f"Registry directory not found: {registry_dir}")
        return

    rebuild_summaries(registry_dir)

if __name__ == "__main__":
    main()
