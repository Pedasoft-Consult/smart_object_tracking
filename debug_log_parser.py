#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug Log Parser for YOLOv5 Training.

This script analyzes YOLOv5 training logs to identify common issues.
"""

import re
import os
import sys
from pathlib import Path


def analyze_training_log(log_file):
    """
    Analyze YOLOv5 training log to identify issues.

    Args:
        log_file: Path to log file or log content as string

    Returns:
        dict: Analysis results
    """
    issues = []
    warnings = []
    observations = []

    # Check if input is a file path or content
    if os.path.exists(log_file):
        print(f"Reading log file: {log_file}")
        with open(log_file, 'r') as f:
            content = f.read()
    else:
        print("Analyzing provided log content")
        content = log_file

    # Look for known issues
    if "Dataset not found" in content:
        path_match = re.search(r"missing paths \[(.*?)\]", content)
        if path_match:
            missing_path = path_match.group(1)
            issues.append(f"Dataset not found: {missing_path}")
            suggestions = [
                "Run the fix_validation_split.py script to create a proper validation set",
                "Ensure both 'images/val' and 'labels/val' directories exist and contain files",
                "Check that the dataset.yaml file has the correct paths"
            ]
            issues.append({"issue": "Missing validation dataset", "suggestions": suggestions})

    if "COMET WARNING" in content:
        warnings.append("Comet ML integration is active but credentials are not set")
        suggestions = ["Set COMET_MODE=disabled in your environment to disable Comet ML"]
        warnings.append({"warning": "Comet ML configuration", "suggestions": suggestions})

    # Check for directory structure in log
    if "[AutoFix] Validation set already exists" in content and "Dataset not found" in content:
        issues.append("Validation directory exists but doesn't have the correct structure or is empty")
        suggestions = [
            "The validation directory exists but might be empty or in the wrong location",
            "Run the fix_validation_split.py script to ensure proper structure",
            "Make sure 'val/images' and 'val/labels' are populated with files"
        ]
        issues.append({"issue": "Invalid validation directory", "suggestions": suggestions})

    # Check for proper YOLOv5 structure references
    if "Error loading metadata" in content or "Failed to clone YOLOv5 repository" in content:
        issues.append("Issues with YOLOv5 repository or metadata")
        suggestions = [
            "Ensure the YOLOv5 repository can be cloned properly",
            "Check network connectivity and Git installation",
            "Try manually cloning the YOLOv5 repository if automatic cloning fails"
        ]
        issues.append({"issue": "YOLOv5 setup problem", "suggestions": suggestions})

    # Look for dataset YAML issues
    if "dataset.yaml" in content:
        yaml_issues = []
        if "dataset YAML not found" in content:
            yaml_issues.append("dataset.yaml file not found")
        if "Error creating dataset YAML" in content:
            yaml_issues.append("Error creating dataset.yaml")

        if yaml_issues:
            issues.append({"issue": "Dataset YAML issues", "details": yaml_issues, "suggestions": [
                "Ensure dataset.yaml exists in the dataset directory",
                "Run the fix_validation_split.py script to regenerate the YAML file",
                "Check that the YAML file has the correct paths and class information"
            ]})

    # Check for training output/progress
    if "MODEL SAVE DETECTED" in content and "Epoch:" not in content:
        observations.append("Training initialized but did not progress to training epochs")

    # Return comprehensive analysis
    return {
        "issues": issues,
        "warnings": warnings,
        "observations": observations,
        "summary": "Found {} issues, {} warnings, and {} observations.".format(
            len([i for i in issues if isinstance(i, str)]) + len([i for i in issues if isinstance(i, dict)]),
            len([w for w in warnings if isinstance(w, str)]) + len([w for w in warnings if isinstance(w, dict)]),
            len(observations)
        )
    }


def print_analysis(analysis):
    """
    Print analysis results in a structured way.

    Args:
        analysis: Analysis results dictionary
    """
    print("\n" + "=" * 80)
    print("YOLOV5 TRAINING LOG ANALYSIS")
    print("=" * 80)

    if analysis["issues"]:
        print("\n🔴 ISSUES:")
        for issue in analysis["issues"]:
            if isinstance(issue, str):
                print(f"  • {issue}")
            elif isinstance(issue, dict):
                print(f"  • {issue.get('issue', 'Issue')}")
                if "details" in issue:
                    for detail in issue["details"]:
                        print(f"    - {detail}")
                if "suggestions" in issue:
                    print("    Suggestions:")
                    for suggestion in issue["suggestions"]:
                        print(f"    → {suggestion}")

    if analysis["warnings"]:
        print("\n🟠 WARNINGS:")
        for warning in analysis["warnings"]:
            if isinstance(warning, str):
                print(f"  • {warning}")
            elif isinstance(warning, dict):
                print(f"  • {warning.get('warning', 'Warning')}")
                if "suggestions" in warning:
                    print("    Suggestions:")
                    for suggestion in warning["suggestions"]:
                        print(f"    → {suggestion}")

    if analysis["observations"]:
        print("\n🔍 OBSERVATIONS:")
        for observation in analysis["observations"]:
            print(f"  • {observation}")

    print("\n" + "-" * 80)
    print("SUMMARY:")
    print(analysis["summary"])
    print("=" * 80)

    print("\n🛠️ NEXT STEPS:")
    print("1. Run the fix_validation_split.py script to fix dataset structure issues")
    print("2. After fixing the structure, try running the training again")
    print("3. If issues persist, check the suggestions for each identified problem")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        analysis = analyze_training_log(log_file)
        print_analysis(analysis)
    else:
        print("Please provide a log file path as an argument.")
        print("Usage: python debug_log_parser.py <log_file>")