#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Exporter for Smart Object Tracking System.
Handles exporting trained models to various formats for deployment.
"""

import os
import logging
import shutil
import subprocess
import sys
import time

import torch
from pathlib import Path


class ModelExporter:
    """Exporter for converting models to deployment formats"""

    def __init__(self, output_dir, config, metadata):
        """
        Initialize model exporter.

        Args:
            output_dir: Path to output directory
            config: Configuration dictionary
            metadata: Model metadata dictionary
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.metadata = metadata
        self.logger = logging.getLogger('ModelExporter')

    def export_model(self, model_id_or_path, format='onnx', output_path=None, img_size=640):
        """
        Export model to specified format.

        Args:
            model_id_or_path: Model ID in registry or direct path to model file
            format: Export format ('onnx', 'tflite', 'trt', etc.)
            output_path: Path to save exported model
            img_size: Input image size for the model

        Returns:
            str: Path to exported model or None if export failed
        """
        try:
            # Determine if input is a model ID or a direct path
            model_path = model_id_or_path
            model_info = None
            self.logger.info(f"Resolved model path: {model_path}")

            # If model_id_or_path looks like a model ID, find it in registry
            if isinstance(model_id_or_path, str) and not os.path.exists(
                    model_id_or_path) and model_id_or_path.startswith("model_"):
                for model in self.metadata["models"]:
                    if model["model_id"] == model_id_or_path:
                        model_info = model
                        model_path = model["model_path"]
                        break

                if model_info is None:
                    self.logger.error(f"Model not found in registry: {model_id_or_path}")
                    return None

            # Check if model path exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return None

            # Check if already exported to requested format
            existing_path = model_info.get(f"{format}_path") if model_info else None
            if existing_path and isinstance(existing_path, (str, bytes, os.PathLike)) and os.path.exists(existing_path):

                self.logger.info(f"Model already exported to {format}: {model_info[f'{format}_path']}")

                # Copy to output path if specified
                if output_path:
                    shutil.copy(model_info[f"{format}_path"], output_path)
                    return output_path

                return model_info[f"{format}_path"]

            # Set default output path if not specified
            if output_path is None:
                model_dir = os.path.dirname(model_path)
                output_path = os.path.join(model_dir, f"model.{format}")

            # Use appropriate export method based on format
            if format.lower() == 'onnx':
                return self._export_to_onnx(model_path, output_path, img_size, model_info)
            elif format.lower() == 'tflite':
                return self._export_to_tflite(model_path, output_path, img_size, model_info)
            elif format.lower() == 'trt':
                return self._export_to_tensorrt(model_path, output_path, img_size, model_info)
            elif format.lower() == 'openvino':
                return self._export_to_openvino(model_path, output_path, img_size, model_info)
            else:
                return self._export_with_yolov5(model_path, format, output_path, img_size, model_info)

        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _ensure_yolov5_repo(self):
        """
        Ensure YOLOv5 repository is available.

        Returns:
            Path: Path to YOLOv5 repository or None if failed
        """
        try:
            # Set up YOLOv5 directory in output directory
            yolov5_dir = self.output_dir / "yolov5"

            # Check if directory exists
            if yolov5_dir.exists():
                # Update repository
                self.logger.info(f"Updating YOLOv5 repository at {yolov5_dir}")
                subprocess.run(
                    ["git", "pull"],
                    cwd=yolov5_dir,
                    check=True,
                    capture_output=True
                )
            else:
                # Clone repository
                self.logger.info(f"Cloning YOLOv5 repository to {yolov5_dir}")
                subprocess.run(
                    ["git", "clone", "https://github.com/ultralytics/yolov5.git", str(yolov5_dir)],
                    check=True,
                    capture_output=True
                )

            # Install dependencies
            self.logger.info("Installing YOLOv5 dependencies")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(yolov5_dir / "requirements.txt")],
                check=True,
                capture_output=True
            )

            return yolov5_dir

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.cmd} (exit code {e.returncode})")
            self.logger.error(f"Error output: {e.stderr}")
            return None

        except Exception as e:
            self.logger.error(f"Error ensuring YOLOv5 repository: {e}")
            return None

    def _export_to_onnx(self, model_path, output_path, img_size, model_info=None):
        """
        Export PyTorch model to ONNX format.

        Args:
            model_path: Path to PyTorch model
            output_path: Path to save ONNX model
            img_size: Input image size
            model_info: Model information dictionary (optional)

        Returns:
            str: Path to ONNX model or None if conversion failed
        """
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return None

            # Use YOLOv5 export script if available
            yolov5_dir = self._ensure_yolov5_repo()
            if yolov5_dir and (yolov5_dir / "export.py").exists():
                self.logger.info(f"Converting model to ONNX using YOLOv5 export script")

                # Run export script
                subprocess.run(
                    [
                        sys.executable,
                        str(yolov5_dir / "export.py"),
                        "--weights", model_path,
                        "--include", "onnx",
                        "--img", str(img_size),
                        "--simplify"
                    ],
                    check=True,
                    capture_output=True
                )

                # Find ONNX file in the same directory as model_path
                model_dir = os.path.dirname(model_path)
                for file in os.listdir(model_dir):
                    if file.endswith(".onnx"):
                        onnx_path = os.path.join(model_dir, file)
                        # Move to output path
                        shutil.move(onnx_path, output_path)

                        # Update model info if available
                        if model_info is not None:
                            model_info["onnx_path"] = output_path

                        return output_path

            # Manual conversion if export script not available
            self.logger.info(f"Converting model to ONNX manually")

            # Load model
            model = torch.load(model_path)
            if isinstance(model, dict) and "model" in model:
                model = model["model"]

            # Set model to evaluation mode
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(1, 3, img_size, img_size)

            # Export model
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                verbose=False,
                opset_version=12,
                input_names=["images"],
                output_names=["output"],
                dynamic_axes={
                    "images": {0: "batch_size"},
                    "output": {0: "batch_size"}
                }
            )

            # Update model info if available
            if model_info is not None:
                model_info["onnx_path"] = output_path

            return output_path

        except Exception as e:
            self.logger.error(f"Error converting model to ONNX: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _export_to_tflite(self, model_path, output_path, img_size, model_info=None):
        """
        Export model to TFLite format via ONNX.

        Args:
            model_path: Path to PyTorch model
            output_path: Path to save TFLite model
            img_size: Input image size
            model_info: Model information dictionary (optional)

        Returns:
            str: Path to TFLite model or None if conversion failed
        """
        try:
            # First convert to ONNX
            onnx_path = self._export_to_onnx(model_path, output_path.replace('.tflite', '.onnx'), img_size)

            if not onnx_path:
                self.logger.error("ONNX conversion failed, cannot proceed to TFLite")
                return None

            # Check if onnx-tf is installed
            try:
                import onnx
                import tf2onnx
                import tensorflow as tf
            except ImportError:
                self.logger.error("Required packages not installed. Install with: pip install onnx onnx-tf tensorflow")
                return None

            # Convert ONNX to TensorFlow SavedModel
            self.logger.info("Converting ONNX to TensorFlow SavedModel")

            # Load ONNX model
            onnx_model = onnx.load(onnx_path)

            # Convert to TensorFlow
            tf_rep = tf2onnx.convert.from_onnx(onnx_model)

            # Save as SavedModel
            saved_model_dir = os.path.dirname(output_path) + "/saved_model"
            os.makedirs(saved_model_dir, exist_ok=True)
            tf_rep.export_graph(saved_model_dir)

            # Convert SavedModel to TFLite
            self.logger.info("Converting SavedModel to TFLite")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

            # Set optimization flags
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Convert the model
            tflite_model = converter.convert()

            # Write model to file
            with open(output_path, 'wb') as f:
                f.write(tflite_model)

            # Clean up intermediate files
            shutil.rmtree(saved_model_dir, ignore_errors=True)

            # Update model info if available
            if model_info is not None:
                model_info["tflite_path"] = output_path

            self.logger.info(f"Model converted to TFLite: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error converting to TFLite: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _export_to_tensorrt(self, model_path, output_path, img_size, model_info=None):
        """
        Export model to TensorRT format via ONNX.

        Args:
            model_path: Path to PyTorch model
            output_path: Path to save TensorRT model
            img_size: Input image size
            model_info: Model information dictionary (optional)

        Returns:
            str: Path to TensorRT model or None if conversion failed
        """
        try:
            # Use YOLOv5 export script for TensorRT
            yolov5_dir = self._ensure_yolov5_repo()
            if yolov5_dir is None:
                self.logger.error("Failed to set up YOLOv5 repository")
                return None

            self.logger.info(f"Converting model to TensorRT using YOLOv5 export script")

            # Run export script
            subprocess.run(
                [
                    sys.executable,
                    str(yolov5_dir / "export.py"),
                    "--weights", model_path,
                    "--include", "engine",
                    "--img", str(img_size),
                    "--device", "0"  # TensorRT requires CUDA
                ],
                check=True,
                capture_output=True
            )

            # Find engine file in the same directory as model_path
            model_dir = os.path.dirname(model_path)
            for file in os.listdir(model_dir):
                if file.endswith(".engine"):
                    engine_path = os.path.join(model_dir, file)
                    # Move to output path
                    shutil.move(engine_path, output_path)

                    # Update model info if available
                    if model_info is not None:
                        model_info["tensorrt_path"] = output_path

                    return output_path

            self.logger.error("TensorRT engine file not found after export")
            return None

        except Exception as e:
            self.logger.error(f"Error converting to TensorRT: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _export_to_openvino(self, model_path, output_path, img_size, model_info=None):
        """
        Export model to OpenVINO format via ONNX.

        Args:
            model_path: Path to PyTorch model
            output_path: Path to save OpenVINO model
            img_size: Input image size
            model_info: Model information dictionary (optional)

        Returns:
            str: Path to OpenVINO model or None if conversion failed
        """
        try:
            # First convert to ONNX
            onnx_path = self._export_to_onnx(model_path, output_path.replace('.xml', '.onnx'), img_size)

            if not onnx_path:
                self.logger.error("ONNX conversion failed, cannot proceed to OpenVINO")
                return None

            # Use OpenVINO Model Optimizer
            try:
                # Try to import openvino
                from openvino.inference_engine import IECore
            except ImportError:
                self.logger.error("OpenVINO not installed. Install with: pip install openvino")
                return None

            # Run model optimizer command
            mo_command = [
                "mo",
                "--input_model", onnx_path,
                "--output_dir", os.path.dirname(output_path),
                "--model_name", os.path.basename(output_path).replace('.xml', '')
            ]

            subprocess.run(mo_command, check=True, capture_output=True)

            # The output will be a .xml file and a .bin file
            xml_path = output_path

            # Check if files were created
            if not os.path.exists(xml_path):
                self.logger.error(f"OpenVINO conversion failed, {xml_path} not found")
                return None

            # Update model info if available
            if model_info is not None:
                model_info["openvino_path"] = xml_path

            self.logger.info(f"Model converted to OpenVINO: {xml_path}")
            return xml_path

        except Exception as e:
            self.logger.error(f"Error converting to OpenVINO: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _export_with_yolov5(self, model_path, format, output_path, img_size, model_info=None):
        """
        Export model using YOLOv5 export.py script.

        Args:
            model_path: Path to PyTorch model
            format: Export format
            output_path: Path to save exported model
            img_size: Input image size
            model_info: Model information dictionary (optional)

        Returns:
            str: Path to exported model or None if conversion failed
        """
        try:
            # Ensure YOLOv5 repository is available
            yolov5_dir = self._ensure_yolov5_repo()
            if yolov5_dir is None:
                self.logger.error("Failed to set up YOLOv5 repository")
                return None

            self.logger.info(f"Exporting model to {format} using YOLOv5 export script")

            # Run export script
            cmd = [
                sys.executable,
                str(yolov5_dir / "export.py"),
                "--weights", model_path,
                "--include", format,
                "--img-size", str(img_size),
                "--batch-size", "1"
            ]

            # Add format-specific options
            if format == 'trt':
                cmd.extend(["--device", "0"])  # TensorRT requires CUDA

            # Run export
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Check for exported file
            model_dir = os.path.dirname(model_path)
            exported_path = None

            # Look for exported file with expected extension
            for file in os.listdir(model_dir):
                if file.endswith(f".{format}"):
                    exported_path = os.path.join(model_dir, file)
                    break

            if exported_path is None:
                self.logger.error(f"Exported file not found after running YOLOv5 export")
                return None

            # Move to output path if different
            if exported_path != output_path:
                shutil.move(exported_path, output_path)

            # Update model info if available
            if model_info is not None:
                model_info[f"{format}_path"] = output_path
                # Update exports record
                if "exports" not in model_info:
                    model_info["exports"] = {}

                model_info["exports"][format] = {
                    "path": output_path,
                    "timestamp": time.time(),
                    "img_size": img_size
                }

            self.logger.info(f"Model exported to {format}: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Export failed: {e.stdout} {e.stderr}")
            return None

        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None