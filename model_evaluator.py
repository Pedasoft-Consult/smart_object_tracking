#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Evaluator for Smart Object Tracking System.
Evaluates model performance on validation/test datasets.
"""
import json
import logging
import subprocess
import sys
import os
import time
from pathlib import Path


class ModelEvaluator:
    """Evaluator for measuring model performance"""

    def __init__(self, output_dir, dataset_dir, config, metadata):
        """
        Initialize model evaluator.

        Args:
            output_dir: Path to output directory
            dataset_dir: Path to dataset directory
            config: Configuration dictionary
            metadata: Model metadata dictionary
        """
        self.output_dir = Path(output_dir)
        self.dataset_dir = Path(dataset_dir)
        self.config = config
        self.metadata = metadata
        self.logger = logging.getLogger('ModelEvaluator')

    def evaluate(self, model_id=None, model_path=None, dataset_yaml=None):
        """
        Evaluate model on validation/test dataset.

        Args:
            model_id: Model ID in registry
            model_path: Direct path to model file (if model_id not provided)
            dataset_yaml: Path to dataset YAML file (defaults to main dataset)

        Returns:
            dict: Evaluation metrics
        """
        try:
            # Determine model path
            if model_id:
                # Find model in registry
                model_info = None
                for model in self.metadata["models"]:
                    if model["model_id"] == model_id:
                        model_info = model
                        break

                if model_info is None:
                    self.logger.error(f"Model not found in registry: {model_id}")
                    return {"error": "Model not found"}

                model_path = model_info["model_path"]

            if not model_path or not os.path.exists(model_path):
                self.logger.error(f"Invalid model path: {model_path}")
                return {"error": "Invalid model path"}

            # Determine dataset YAML
            if dataset_yaml is None:
                dataset_yaml = str(self.dataset_dir / "dataset.yaml")

            if not os.path.exists(dataset_yaml):
                self.logger.error(f"Dataset YAML not found: {dataset_yaml}")
                return {"error": "Dataset YAML not found"}

            # Use YOLOv5 validation script
            metrics = self._evaluate_with_yolov5(model_path, dataset_yaml)

            # Store evaluation results if model_id was provided
            if model_id and metrics and "error" not in metrics:
                for i, model in enumerate(self.metadata["models"]):
                    if model["model_id"] == model_id:
                        if "evaluation" not in self.metadata["models"][i]:
                            self.metadata["models"][i]["evaluation"] = {}

                        self.metadata["models"][i]["evaluation"]["latest"] = {
                            "timestamp": metrics.get("timestamp", 0),
                            "metrics": metrics
                        }

                        # Store evaluation history
                        if "history" not in self.metadata["models"][i]["evaluation"]:
                            self.metadata["models"][i]["evaluation"]["history"] = []

                        self.metadata["models"][i]["evaluation"]["history"].append({
                            "timestamp": metrics.get("timestamp", 0),
                            "metrics": metrics
                        })

                        # Save metadata
                        try:
                            with open(self.output_dir / "training_metadata.json", 'w') as f:
                                json.dump(self.metadata, f, indent=2)
                        except Exception as e:
                            self.logger.error(f"Error saving metadata: {e}")

                        break

            return metrics

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

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

    def _evaluate_with_yolov5(self, model_path, dataset_yaml):
        """
        Evaluate model using YOLOv5 val.py script.

        Args:
            model_path: Path to model file
            dataset_yaml: Path to dataset YAML file

        Returns:
            dict: Evaluation metrics
        """
        try:
            # Set up YOLOv5 directory
            yolov5_dir = self._ensure_yolov5_repo()
            if yolov5_dir is None:
                self.logger.error("Failed to set up YOLOv5 repository")
                return {"error": "Failed to set up YOLOv5 repository"}

            # Run evaluation
            self.logger.info(f"Evaluating model {model_path} on dataset {dataset_yaml}")
            import time
            eval_output = os.path.join(self.output_dir, f"eval_{int(time.time())}")
            os.makedirs(eval_output, exist_ok=True)
            print(eval_output)
            # Set up validation command
            cmd = [
                sys.executable,
                str(yolov5_dir / "val.py"),
                "--data", dataset_yaml,
                "--weights", model_path,
                "--project", eval_output,
                "--name", "eval",
                "--verbose",
                "--save-json"  # Save results in JSON format
            ]

            # Run validation
            import time
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            evaluation_time = time.time() - start_time

            # Parse output
            metrics = {
                "timestamp": time.time(),
                "evaluation_time": evaluation_time
            }

            for line in result.stdout.splitlines():
                if "mAP@0.5" in line:
                    try:
                        metrics["mAP_0.5"] = float(line.split("mAP@0.5 =")[1].split()[0])
                    except:
                        pass

                if "mAP@0.5:0.95" in line:
                    try:
                        metrics["mAP_0.5_0.95"] = float(line.split("mAP@0.5:0.95 =")[1].split()[0])
                    except:
                        pass

                # Extract precision and recall
                if "Precision:" in line:
                    try:
                        metrics["precision"] = float(line.split("Precision:")[1].split()[0])
                    except:
                        pass

                if "Recall:" in line:
                    try:
                        metrics["recall"] = float(line.split("Recall:")[1].split()[0])
                    except:
                        pass

                # Extract class-specific metrics
                if "Class" in line and "Images" in line and "Instances" in line:
                    # This line contains the header for class-specific metrics
                    class_metrics_header = True
                    metrics["class_metrics"] = {}

                if "class_metrics" in metrics and "all" in line.lower():
                    # This line contains the overall class metrics
                    try:
                        parts = line.split()
                        class_name = parts[0].lower()
                        metrics["class_metrics"][class_name] = {
                            "images": int(parts[1]),
                            "instances": int(parts[2]),
                            "precision": float(parts[3]),
                            "recall": float(parts[4]),
                            "mAP_0.5": float(parts[5]),
                            "mAP_0.5_0.95": float(parts[6])
                        }
                    except:
                        pass

            # Read additional metrics from JSON if available
            metrics_json = os.path.join(eval_output, "eval", "metrics.json")
            if os.path.exists(metrics_json):
                try:
                    with open(metrics_json, 'r') as f:
                        json_metrics = json.load(f)
                        metrics.update(json_metrics)
                except Exception as e:
                    self.logger.warning(f"Error reading metrics JSON: {e}")

            # Read confusion matrix if available
            confusion_matrix_path = os.path.join(eval_output, "eval", "confusion_matrix.png")
            if os.path.exists(confusion_matrix_path):
                metrics["confusion_matrix_path"] = confusion_matrix_path

            self.logger.info(f"Evaluation completed: mAP@0.5={metrics.get('mAP_0.5', 'N/A')}, "
                             f"mAP@0.5:0.95={metrics.get('mAP_0.5_0.95', 'N/A')}")

            return metrics

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Validation failed: {e.stderr}")
            return {"error": f"Validation failed: {e.stderr}"}

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def compare_models(self, model_ids):
        """
        Compare performance of multiple models.

        Args:
            model_ids: List of model IDs to compare

        Returns:
            dict: Comparison results
        """
        try:
            # Collect evaluation metrics for all models
            comparison = {
                "models": [],
                "metrics": ["mAP_0.5", "mAP_0.5_0.95", "precision", "recall"],
                "timestamp": time.time()
            }

            for model_id in model_ids:
                # Find model in registry
                model_info = None
                for model in self.metadata["models"]:
                    if model["model_id"] == model_id:
                        model_info = model
                        break

                if model_info is None:
                    self.logger.warning(f"Model not found in registry: {model_id}")
                    continue

                # Check if model has evaluation metrics
                if "evaluation" not in model_info or "latest" not in model_info["evaluation"]:
                    # Evaluate model first
                    evaluation = self.evaluate(model_id)

                    if "error" in evaluation:
                        self.logger.warning(f"Failed to evaluate model {model_id}: {evaluation['error']}")
                        continue

                    metrics = evaluation
                else:
                    metrics = model_info["evaluation"]["latest"]["metrics"]

                # Add to comparison
                comparison["models"].append({
                    "model_id": model_id,
                    "model_type": model_info.get("model_type", "unknown"),
                    "created": model_info.get("created", 0),
                    "metrics": {
                        metric: metrics.get(metric, 0) for metric in comparison["metrics"]
                    }
                })

            # Determine best model for each metric
            for metric in comparison["metrics"]:
                best_value = -1
                best_model = None

                for model in comparison["models"]:
                    value = model["metrics"].get(metric, 0)
                    if value > best_value:
                        best_value = value
                        best_model = model["model_id"]

                comparison[f"best_{metric}_model"] = best_model
                comparison[f"best_{metric}_value"] = best_value

            # Determine overall best model (based on mAP_0.5_0.95)
            if comparison["models"]:
                comparison["best_overall_model"] = comparison.get("best_mAP_0.5_0.95_model")

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def analyze_model_performance(self, model_id):
        """
        Perform detailed analysis of model performance.

        Args:
            model_id: Model ID to analyze

        Returns:
            dict: Detailed performance analysis
        """
        try:
            # Find model in registry
            model_info = None
            for model in self.metadata["models"]:
                if model["model_id"] == model_id:
                    model_info = model
                    break

            if model_info is None:
                self.logger.error(f"Model not found in registry: {model_id}")
                return {"error": "Model not found"}

            # Evaluate model if no evaluation available
            if "evaluation" not in model_info:
                evaluation = self.evaluate(model_id)

                if "error" in evaluation:
                    return {"error": f"Evaluation failed: {evaluation['error']}"}

                metrics = evaluation
            else:
                metrics = model_info["evaluation"]["latest"]["metrics"]

            # Get dataset statistics
            dataset_stats = model_info.get("dataset_stats", {})

            # Create performance analysis
            analysis = {
                "model_id": model_id,
                "model_type": model_info.get("model_type", "unknown"),
                "created": model_info.get("created", 0),
                "analysis_timestamp": time.time(),
                "metrics": metrics,
                "dataset_stats": dataset_stats,
                "performance_summary": {},
                "improvement_suggestions": []
            }

            # Generate performance summary
            mAP_0_5 = metrics.get("mAP_0.5", 0)
            mAP_0_5_0_95 = metrics.get("mAP_0.5_0.95", 0)
            precision = metrics.get("precision", 0)
            recall = metrics.get("recall", 0)

            # Overall performance rating
            if mAP_0_5 >= 0.9:
                rating = "Excellent"
            elif mAP_0_5 >= 0.8:
                rating = "Very Good"
            elif mAP_0_5 >= 0.7:
                rating = "Good"
            elif mAP_0_5 >= 0.5:
                rating = "Fair"
            else:
                rating = "Poor"

            analysis["performance_summary"]["rating"] = rating
            analysis["performance_summary"]["mAP_0.5"] = mAP_0_5
            analysis["performance_summary"]["mAP_0.5_0.95"] = mAP_0_5_0_95
            analysis["performance_summary"]["precision"] = precision
            analysis["performance_summary"]["recall"] = recall

            # Generate improvement suggestions
            if mAP_0_5 < 0.7:
                if dataset_stats.get("train_images", 0) < 1000:
                    analysis["improvement_suggestions"].append("Increase training dataset size")
                analysis["improvement_suggestions"].append("Try a larger model architecture")
                analysis["improvement_suggestions"].append("Increase training epochs")

            if precision < 0.7:
                analysis["improvement_suggestions"].append("Increase confidence threshold for inference")
                analysis["improvement_suggestions"].append("Improve data quality and annotations")

            if recall < 0.7:
                analysis["improvement_suggestions"].append("Decrease confidence threshold for inference")
                analysis["improvement_suggestions"].append("Add more diverse training examples")

            # Class-specific analysis
            if "class_metrics" in metrics:
                analysis["class_analysis"] = {}

                for class_name, class_metrics in metrics["class_metrics"].items():
                    if class_name == "all":
                        continue

                    class_mAP = class_metrics.get("mAP_0.5", 0)

                    # Determine class performance
                    if class_mAP >= 0.9:
                        class_performance = "Excellent"
                    elif class_mAP >= 0.8:
                        class_performance = "Very Good"
                    elif class_mAP >= 0.7:
                        class_performance = "Good"
                    elif class_mAP >= 0.5:
                        class_performance = "Fair"
                    else:
                        class_performance = "Poor"

                    analysis["class_analysis"][class_name] = {
                        "performance": class_performance,
                        "metrics": class_metrics,
                        "suggestions": []
                    }

                    # Add class-specific suggestions
                    if class_mAP < 0.7:
                        analysis["class_analysis"][class_name]["suggestions"].append(
                            f"Add more training examples for class '{class_name}'"
                        )
                        analysis["class_analysis"][class_name]["suggestions"].append(
                            f"Check annotation quality for class '{class_name}'"
                        )

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing model performance: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def get_evaluation_history(self, model_id):
        """
        Get evaluation history for a model.

        Args:
            model_id: Model ID to get history for

        Returns:
            list: List of historical evaluation results
        """
        try:
            # Find model in registry
            for model in self.metadata["models"]:
                if model["model_id"] == model_id:
                    # Check if model has evaluation history
                    if "evaluation" in model and "history" in model["evaluation"]:
                        return model["evaluation"]["history"]
                    else:
                        return []

            return {"error": f"Model not found: {model_id}"}

        except Exception as e:
            self.logger.error(f"Error getting evaluation history: {e}")
            return {"error": str(e)}