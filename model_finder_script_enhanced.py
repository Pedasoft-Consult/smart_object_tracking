#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Model Finder - Script to locate trained model files
This script helps find model files that might be in unexpected locations
"""

import os
import sys
import glob
import argparse
import time
from pathlib import Path
import subprocess


def find_model_files(base_dir, training_id=None, extensions=None, hours=24, verbose=True):
    """
    Find model files in the directory and its subdirectories

    Args:
        base_dir: Base directory to search from
        training_id: Optional training ID to filter results
        extensions: List of file extensions to look for (defaults to ['.pt'])
        hours: Only include files modified in the last X hours
        verbose: Whether to print detailed information

    Returns:
        List of found model files
    """
    if extensions is None:
        extensions = ['.pt']

    # Convert to Path object if it's a string
    base_dir = Path(base_dir)

    if verbose:
        print(f"Searching for model files in: {base_dir}")
        print(f"Looking for extensions: {extensions}")
        if training_id:
            print(f"Filtering for training ID: {training_id}")
        print(f"Time window: files modified in the last {hours} hours")

    # Check if directory exists
    if not base_dir.exists():
        print(f"ERROR: Directory {base_dir} does not exist!")
        return []

    # List all files in directory and subdirectories
    found_files = []

    # Calculate cutoff time
    cutoff_time = time.time() - (hours * 3600)

    # Method 1: Use recursive glob
    for ext in extensions:
        pattern = f"**/*{ext}"
        for file in base_dir.glob(pattern):
            if file.is_file():
                # Check if file matches training ID
                if training_id and training_id not in str(file):
                    continue

                # Check modification time
                mod_time = file.stat().st_mtime
                if mod_time >= cutoff_time:
                    found_files.append((file, mod_time, file.stat().st_size))

    # Method 2: Use os.walk for more thorough search (sometimes glob misses files)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = Path(root) / file

                # Check if file matches training ID
                if training_id and training_id not in str(file_path):
                    continue

                # Check modification time
                try:
                    mod_time = file_path.stat().st_mtime
                    if mod_time >= cutoff_time:
                        found_files.append((file_path, mod_time, file_path.stat().st_size))
                except Exception as e:
                    print(f"Error accessing {file_path}: {e}")

    # Method 3: Use find command (on Linux/Mac)
    try:
        cmd = ["find", str(base_dir), "-type", "f"]

        # Add extension pattern
        ext_pattern = " -o ".join([f"-name '*{ext}'" for ext in extensions])
        cmd.extend(["-o", ext_pattern])

        # Add time filter
        cmd.extend(["-mtime", f"-{hours}h"])

        # Execute find command
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            for line in result.stdout.splitlines():
                file_path = Path(line.strip())
                if file_path.exists():
                    # Check if file matches training ID
                    if training_id and training_id not in str(file_path):
                        continue

                    mod_time = file_path.stat().st_mtime
                    found_files.append((file_path, mod_time, file_path.stat().st_size))
    except Exception as e:
        # The find command might not work on all platforms, so ignore errors
        pass

    # Remove duplicates (convert to set and back to list)
    unique_files = []
    paths_seen = set()
    for file_path, mod_time, size in found_files:
        if str(file_path) not in paths_seen:
            paths_seen.add(str(file_path))
            unique_files.append((file_path, mod_time, size))

    # Sort by modification time (newest first)
    unique_files.sort(key=lambda x: x[1], reverse=True)

    # Format for display
    result_files = []
    for file_path, mod_time, size in unique_files:
        size_mb = size / (1024 * 1024)  # Size in MB
        result_files.append({
            "path": file_path,
            "modified": mod_time,
            "size_mb": size_mb,
            "modified_time": time.ctime(mod_time)
        })

    # Print results
    if verbose:
        print(f"\nFound {len(result_files)} model files:")
        for i, file_info in enumerate(result_files):
            print(f"  {i + 1}. {file_info['path']} ({file_info['size_mb']:.2f} MB)")
            print(f"     Modified: {file_info['modified_time']}")

    return result_files


def check_yolov5_output_structure(base_dir, training_id):
    """
    Analyze the YOLOv5 output directory structure

    Args:
        base_dir: Base directory
        training_id: Training ID
    """
    base_dir = Path(base_dir)

    # Check multiple possible locations for YOLOv5 output
    possible_train_dirs = [
        base_dir / training_id,
        base_dir / "yolov5" / "runs" / "train" / training_id,
        base_dir / "runs" / "train" / training_id
    ]

    found = False
    for train_dir in possible_train_dirs:
        if train_dir.exists():
            found = True
            print(f"\nFound YOLOv5 output directory: {train_dir}")

            # Check for common YOLOv5 output directories and files
            expected_items = {
                "weights": "directory where models are typically saved",
                "weights/best.pt": "best model checkpoint",
                "weights/last.pt": "last model checkpoint",
                "results.png": "training results plot",
                "train_batch0.jpg": "training batch visualization",
                "val_batch0_pred.jpg": "validation batch predictions"
            }

            print("\nChecking for expected YOLOv5 output files/directories:")
            for item, description in expected_items.items():
                item_path = train_dir / item
                status = "EXISTS" if item_path.exists() else "NOT FOUND"
                if item_path.exists() and item_path.is_file():
                    size_mb = item_path.stat().st_size / (1024 * 1024)
                    status += f" ({size_mb:.2f} MB)"
                print(f"  {item}: {status} - {description}")

            # List all directories
            print("\nDirectory contents:")
            try:
                for item in sorted(os.listdir(train_dir)):
                    item_path = train_dir / item
                    type_str = "Directory" if item_path.is_dir() else "File"
                    size_str = ""
                    if item_path.is_file():
                        size_mb = item_path.stat().st_size / (1024 * 1024)
                        size_str = f" ({size_mb:.2f} MB)"
                    print(f"  {item} ({type_str}){size_str}")

                    # If it's weights directory, list contents
                    if item == "weights" and item_path.is_dir():
                        print("    Contents of weights directory:")
                        for weight_file in sorted(os.listdir(item_path)):
                            weight_path = item_path / weight_file
                            if weight_path.is_file():
                                size_mb = weight_path.stat().st_size / (1024 * 1024)
                                print(f"      {weight_file} ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"  Error reading directory: {e}")

    if not found:
        print(f"\nNo YOLOv5 output directory found for training ID: {training_id}")
        print("Checked the following locations:")
        for dir_path in possible_train_dirs:
            print(f"  {dir_path}")


def check_comet_ml(base_dir):
    """
    Check for Comet ML files that might contain model information

    Args:
        base_dir: Base directory
    """
    base_dir = Path(base_dir)
    comet_dir = base_dir / ".cometml-runs"

    if comet_dir.exists():
        print(f"\nFound Comet ML directory: {comet_dir}")

        # List zip files
        zip_files = list(comet_dir.glob("*.zip"))

        if zip_files:
            print(f"Found {len(zip_files)} Comet ML archive files:")
            for i, zip_file in enumerate(sorted(zip_files, key=lambda x: x.stat().st_mtime, reverse=True)):
                mod_time = time.ctime(zip_file.stat().st_mtime)
                size_mb = zip_file.stat().st_size / (1024 * 1024)
                print(f"  {i + 1}. {zip_file.name} (Modified: {mod_time}, Size: {size_mb:.2f} MB)")

            print("\nComet ML might be storing model files. Try:")
            print(f"comet upload {zip_files[0]}")
            print("Then check the experiment in the Comet ML web interface.")
        else:
            print("No Comet ML archive files found.")
    else:
        print("\nNo Comet ML directory found.")


def check_file_permissions(base_dir):
    """
    Check file permissions in the output directory

    Args:
        base_dir: Base directory
    """
    base_dir = Path(base_dir)

    print(f"\nChecking file permissions in: {base_dir}")

    try:
        # Check if we can write to the directory
        test_file = base_dir / "permission_test.txt"
        try:
            with open(test_file, 'w') as f:
                f.write("Testing write permissions")
            print("✓ Write permission test: SUCCESS")
            # Clean up test file
            os.remove(test_file)
        except Exception as e:
            print(f"✗ Write permission test: FAILED - {e}")

        # Check directory permissions
        try:
            import stat
            mode = base_dir.stat().st_mode
            permission_string = stat.filemode(mode)
            print(f"Directory permissions: {permission_string}")

            # Check owner and group
            import pwd
            import grp
            stat_info = os.stat(base_dir)
            uid = stat_info.st_uid
            gid = stat_info.st_gid
            user = pwd.getpwuid(uid).pw_name
            group = grp.getgrgid(gid).gr_name
            print(f"Directory owner: {user}:{group}")

            # Check current user
            import getpass
            current_user = getpass.getuser()
            print(f"Current user: {current_user}")

            if current_user != user:
                print(f"WARNING: You are running as {current_user} but the directory is owned by {user}")
        except Exception as e:
            print(f"Error checking permissions: {e}")

        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(base_dir)
            print(f"Disk space: {free / (1024 ** 3):.1f} GB free of {total / (1024 ** 3):.1f} GB total")

            if free < 1024 ** 3:  # Less than 1 GB
                print("WARNING: Low disk space might be causing issues")
        except Exception as e:
            print(f"Error checking disk space: {e}")

    except Exception as e:
        print(f"Error during permission checks: {e}")


def check_yolov5_version(base_dir):
    """
    Check YOLOv5 version

    Args:
        base_dir: Base directory where YOLOv5 is cloned
    """
    base_dir = Path(base_dir)
    possible_yolov5_dirs = [
        base_dir / "yolov5",
        base_dir / "models" / "yolov5"
    ]

    for yolov5_dir in possible_yolov5_dirs:
        if yolov5_dir.exists():
            print(f"\nFound YOLOv5 directory: {yolov5_dir}")

            # Try to get version from git
            try:
                os.chdir(yolov5_dir)
                result = subprocess.run(["git", "log", "-1", "--format=%h %cd"],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"YOLOv5 version (git): {result.stdout.strip()}")

                # Check branch
                result = subprocess.run(["git", "branch", "--show-current"],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"YOLOv5 branch: {result.stdout.strip()}")

                # Check for local modifications
                result = subprocess.run(["git", "status", "-s"],
                                        capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    print("WARNING: YOLOv5 has local modifications:")
                    print(result.stdout)
            except Exception as e:
                print(f"Error checking git info: {e}")

            # Check requirements.txt
            req_file = yolov5_dir / "requirements.txt"
            if req_file.exists():
                print("\nChecking YOLOv5 requirements:")
                try:
                    with open(req_file, 'r') as f:
                        reqs = f.readlines()

                    for req in reqs:
                        if "torch" in req or "comet" in req or "matplotlib" in req:
                            print(f"  {req.strip()}")
                except Exception as e:
                    print(f"Error reading requirements.txt: {e}")

            # Check if there's a custom train.py
            train_py = yolov5_dir / "train.py"
            if train_py.exists():
                print("\nChecking YOLOv5 train.py for save-related code:")
                try:
                    with open(train_py, 'r') as f:
                        lines = f.readlines()

                    # Look for save-related lines
                    save_lines = []
                    for i, line in enumerate(lines):
                        if "save" in line.lower() and "model" in line.lower():
                            context_start = max(0, i - 2)
                            context_end = min(len(lines), i + 3)
                            context = "".join(lines[context_start:context_end])
                            save_lines.append((i + 1, context))

                    if save_lines:
                        print(f"Found {len(save_lines)} model save locations in train.py:")
                        for line_num, context in save_lines[:5]:  # Show first 5
                            print(f"  Line {line_num}:")
                            print("    " + context.replace("\n", "\n    "))

                        if len(save_lines) > 5:
                            print(f"  ... and {len(save_lines) - 5} more.")
                    else:
                        print("No obvious model save code found in train.py")
                except Exception as e:
                    print(f"Error analyzing train.py: {e}")

            return

    print("\nNo YOLOv5 directory found")


def provide_fix_suggestions(model_files, training_id):
    """
    Provide suggestions to fix the model loading issue

    Args:
        model_files: List of found model files
        training_id: Training ID
    """
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if model_files:
        newest_model = model_files[0]
        print(f"\n1. FOUND MODEL FILE: {newest_model['path']}")
        print("   You can manually copy this file to the expected location:")
        print(f"   cp \"{newest_model['path']}\" models/{training_id}/weights/best.pt")

        print("\n2. UPDATE _register_trained_model METHOD:")
        print("   Modify the method to search in more locations, including:")
        print(f"   - {os.path.dirname(newest_model['path'])}")

        print("\n3. MODIFY YOLOv5 CALL:")
        print("   Add these flags to your YOLOv5 command:")
        print("   --save-period 1 --exist-ok --nosave False")

        print("\n4. DISABLE COMET ML:")
        print("   Add this before running YOLOv5:")
        print("   os.environ['COMET_MODE'] = 'disabled'")

        # Provide a unified fix
        print("\n5. AUTOMATIC FIX SCRIPT:")
        print("   Here's a simple script to copy the found model to the expected location:")
        print("""
   # Copy model file to expected location
   import os
   import shutil
   from pathlib import Path

   source = "{}"
   target = "models/{}/weights/best.pt"

   os.makedirs(os.path.dirname(target), exist_ok=True)
   shutil.copy2(source, target)
   print(f"Copied {source} to {target}")
   """.format(newest_model['path'], training_id))
    else:
        print("\nNO MODEL FILES FOUND! Possible issues:")
        print("  1. Training may have failed to save the model files")
        print("  2. Files might be in a completely different location")
        print("  3. Comet ML might be intercepting the save process")

        print("\nTry these steps:")
        print("  1. Run YOLOv5 directly with --verbose flag")
        print("  2. Disable Comet ML integration")
        print("  3. Check for permission issues in the output directory")
        print("  4. Modify YOLOv5 training code to add debugging statements:")
        print("""
   # Add to YOLOv5 train.py
   print(f"DEBUG: Saving model to {model_save_path}")
   torch.save(model, model_save_path)
   print(f"DEBUG: Model saved successfully: {os.path.exists(model_save_path)}")
   """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find trained model files")
    parser.add_argument("base_dir", help="Base directory to search from")
    parser.add_argument("--training-id", help="Training ID to filter results")
    parser.add_argument("--extensions", nargs="+", default=[".pt"],
                        help="File extensions to search for (default: .pt)")
    parser.add_argument("--hours", type=int, default=24,
                        help="Only include files modified in the last X hours")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze YOLOv5 output structure")
    parser.add_argument("--check-comet", action="store_true",
                        help="Check for Comet ML files")
    parser.add_argument("--check-permissions", action="store_true",
                        help="Check file permissions")
    parser.add_argument("--check-yolov5", action="store_true",
                        help="Check YOLOv5 version and modifications")
    parser.add_argument("--full-analysis", action="store_true",
                        help="Run all checks")

    args = parser.parse_args()

    # Full analysis enables all checks
    if args.full_analysis:
        args.analyze = True
        args.check_comet = True
        args.check_permissions = True
        args.check_yolov5 = True

    # Find model files
    model_files = find_model_files(
        args.base_dir,
        args.training_id,
        args.extensions,
        args.hours
    )

    # Analyze directory structure if requested
    if args.analyze and args.training_id:
        check_yolov5_output_structure(args.base_dir, args.training_id)

    # Check for Comet ML files
    if args.check_comet:
        check_comet_ml(args.base_dir)

    # Check file permissions
    if args.check_permissions:
        check_file_permissions(args.base_dir)

    # Check YOLOv5 version
    if args.check_yolov5:
        check_yolov5_version(args.base_dir)

    # Provide fix recommendations
    provide_fix_suggestions(model_files, args.training_id)

    # Print usage examples
    if not (args.analyze or args.check_comet or args.check_permissions or args.check_yolov5):
        print("\nFor more detailed analysis, try:")
        print(
            f"python {sys.argv[0]} {args.base_dir} --training-id {args.training_id or 'YOUR_TRAINING_ID'} --full-analysis")