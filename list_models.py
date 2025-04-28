#!/usr/bin/env python3
# list_models.py

import os
import json
from pathlib import Path
from tabulate import tabulate

def load_summaries(registry_dir):
    summaries = []
    for model_dir in Path(registry_dir).glob("model_*"):
        summary_file = model_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                    summaries.append(summary)
            except Exception as e:
                print(f"Failed to load {summary_file}: {e}")
    return summaries

def display_table(summaries):
    table = []
    for summary in summaries:
        model_id = summary.get("model_id", "N/A")
        evaluation = summary.get("evaluation", {})
        mAP50 = evaluation.get("mAP_50", "N/A")
        precision = evaluation.get("precision", "N/A")
        recall = evaluation.get("recall", "N/A")
        created_at = summary.get("created_at", "N/A")

        table.append([model_id, mAP50, precision, recall, created_at])

    headers = ["Model ID", "mAP@0.5", "Precision", "Recall", "Created At"]
    print(tabulate(table, headers=headers, tablefmt="grid"))

def main():
    registry_dir = "models/registry"
    if not os.path.exists(registry_dir):
        print(f"Registry directory not found: {registry_dir}")
        return

    summaries = load_summaries(registry_dir)
    if not summaries:
        print("No model summaries found.")
    else:
        display_table(summaries)

if __name__ == "__main__":
    main()
