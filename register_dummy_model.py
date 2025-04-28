# register_dummy_model.py
import time
import os
import json
from pathlib import Path

output_dir = Path("models")
metadata_path = output_dir / "training_metadata.json"

model_id = f"model_{int(time.time())}_yolov5s"
model_path = str(output_dir / "dummy_model.pt")

# Touch dummy model file
Path(model_path).touch()

dummy_entry = {
    "model_id": model_id,
    "training_id": "dummy_training",
    "model_type": "yolov5s",
    "model_path": model_path,
    "onnx_path": None,
    "created": time.time(),
    "dataset_stats": {},
    "training_config": {}
}

with open(metadata_path, "r+") as f:
    metadata = json.load(f)
    metadata.setdefault("models", []).append(dummy_entry)
    f.seek(0)
    json.dump(metadata, f, indent=2)
    f.truncate()

print(f"Registered dummy model: {model_id}")
