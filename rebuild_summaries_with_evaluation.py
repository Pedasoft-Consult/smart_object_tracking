#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Dummy evaluation function
def evaluate_model_dummy(onnx_path):
    # For now, simulate evaluation
    # Later we can improve this with real validation
    return {
        "mAP_50": round(np.random.uniform(0.3, 0.7), 4),
        "mAP_50_95": round(np.random.uniform(0.2, 0.6), 4),
        "precision": round(np.random.uniform(0.3, 0.7), 4),
        "recall": round(np.random.uniform(0.3, 0.7), 4)
    }

def rebuild_summaries_with_eval(registry_dir):
    for model_dir in Path(registry_dir).glob("model_*"):
        summary_file = model_dir / "summary.json"
        model_file = model_dir / "model.onnx"

        if model_file.exists():
            print(f"Evaluating {model_dir.name}...")
            eval_results = evaluate_model_dummy(str(model_file))  # Simulate for now

            if eval_results:
                if summary_file.exists():
                    with open(summary_file, "r") as f:
                        summary = json.load(f)
                else:
                    summary = {}

                summary["evaluation"] = eval_results
                summary.setdefault("training_id", "unknown")
                summary.setdefault("model_id", model_dir.name)
                summary.setdefault("model_type", "unknown")
                summary.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
                summary.setdefault("exported_model_path", str(model_file))

                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)

                print(f"✅ Updated summary for {model_dir.name}")
            else:
                print(f"⚠️ Skipped {model_dir.name} (evaluation failed)")
        else:
            print(f"⚠️ No ONNX model found in {model_dir}")

def main():
    registry_dir = "models/registry"
    if not os.path.exists(registry_dir):
        print(f"Registry directory not found: {registry_dir}")
        return

    rebuild_summaries_with_eval(registry_dir)

if __name__ == "__main__":
    main()
