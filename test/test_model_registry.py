#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the model registry system.
"""

import sys
import os
import pytest
import tempfile
import shutil
import json
import time
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import module to test
from model_registry import ModelRegistry, ModelType


# Mock model files for testing
@pytest.fixture
def mock_model_files():
    """Create temporary mock model files for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model directory
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create mock model files
        models = {
            "pytorch": models_dir / "model.pt",
            "onnx": models_dir / "model.onnx",
            "tensorflow": models_dir / "model.pb",
            "tensorrt": models_dir / "model.engine",
            "tflite": models_dir / "model.tflite",
            "openvino": models_dir / "model.xml",
            "custom": models_dir / "model.custom"
        }

        # Write some test data to each file
        for name, path in models.items():
            path.write_text(f"Test model data for {name}")

            # For OpenVINO, also create the .bin file
            if name == "openvino":
                bin_path = path.with_suffix(".bin")
                bin_path.write_text("OpenVINO weights data")

        yield temp_dir, models


# Test registry initialization
def test_registry_init(mock_model_files):
    """Test that the registry initializes correctly"""
    temp_dir, _ = mock_model_files

    # Create config
    config = {
        "models": {
            "directory": str(Path(temp_dir) / "models")
        }
    }

    # Initialize registry
    registry = ModelRegistry(config)

    # Check it initialized correctly
    assert registry is not None
    assert registry.models_dir == Path(temp_dir) / "models"
    assert registry.registry_file == Path(temp_dir) / "models" / "registry.json"
    assert len(registry.registry["models"]) == 0  # No models registered yet
    assert registry.loaded_models == {}


# Test model type detection
def test_detect_model_type(mock_model_files):
    """Test detection of model type from file extension"""
    temp_dir, models = mock_model_files

    # Create config
    config = {
        "models": {
            "directory": str(Path(temp_dir) / "models")
        }
    }

    # Initialize registry
    registry = ModelRegistry(config)

    # Test type detection for each model
    assert registry._detect_model_type(models["pytorch"]) == ModelType.PYTORCH
    assert registry._detect_model_type(models["onnx"]) == ModelType.ONNX
    assert registry._detect_model_type(models["tensorflow"]) == ModelType.TENSORFLOW
    assert registry._detect_model_type(models["tensorrt"]) == ModelType.TENSORRT
    assert registry._detect_model_type(models["tflite"]) == ModelType.TFLITE
    assert registry._detect_model_type(models["openvino"]) == ModelType.OPENVINO
    assert registry._detect_model_type(models["custom"]) == ModelType.CUSTOM


# Test model registration
def test_register_model(mock_model_files):
    """Test registering a model"""
    temp_dir, models = mock_model_files

    # Create config
    config = {
        "models": {
            "directory": str(Path(temp_dir) / "models")
        }
    }

    # Initialize registry
    registry = ModelRegistry(config)

    # Register a PyTorch model
    model_id = registry.register_model(
        model_path=models["pytorch"],
        name="Test PyTorch Model",
        description="A test model",
        tags=["test", "pytorch"],
        is_default=True
    )

    # Check that it was registered correctly
    assert model_id is not None
    assert len(registry.registry["models"]) == 1
    assert registry.registry["models"][0]["name"] == "Test PyTorch Model"
    assert registry.registry["models"][0]["type"] == "pytorch"
    assert registry.registry["models"][0]["is_default"] == True
    assert "test" in registry.registry["models"][0]["tags"]

    # Register another model
    model_id2 = registry.register_model(
        model_path=models["onnx"],
        name="Test ONNX Model"
    )

    # Check that it was registered and first model is still default
    assert model_id2 is not None
    assert len(registry.registry["models"]) == 2
    assert registry.registry["models"][1]["name"] == "Test ONNX Model"
    assert registry.registry["models"][1]["type"] == "onnx"
    assert registry.registry["models"][1]["is_default"] == False
    assert registry.registry["models"][0]["is_default"] == True

    # Set second model as default
    result = registry.set_default_model(model_id2)
    assert result is True
    assert registry.registry["models"][1]["is_default"] == True
    assert registry.registry["models"][0]["is_default"] == False


# Test model unregistration
def test_unregister_model(mock_model_files):
    """Test unregistering a model"""
    temp_dir, models = mock_model_files

    # Create config
    config = {
        "models": {
            "directory": str(Path(temp_dir) / "models")
        }
    }

    # Initialize registry
    registry = ModelRegistry(config)

    # Register two models
    model_id1 = registry.register_model(models["pytorch"], "Model 1")
    model_id2 = registry.register_model(models["onnx"], "Model 2")

    # Unregister first model
    result = registry.unregister_model(model_id1)
    assert result is True
    assert len(registry.registry["models"]) == 1
    assert registry.registry["models"][0]["id"] == model_id2

    # Unregister model that doesn't exist
    result = registry.unregister_model("nonexistent_id")
    assert result is False


# Test listing models
def test_list_models(mock_model_files):
    """Test listing models with filters"""
    temp_dir, models = mock_model_files

    # Create config
    config = {
        "models": {
            "directory": str(Path(temp_dir) / "models")
        }
    }

    # Initialize registry
    registry = ModelRegistry(config)

    # Register several models with different types and tags
    registry.register_model(models["pytorch"], "PyTorch Model 1", tags=["small", "fast"])
    registry.register_model(models["pytorch"], "PyTorch Model 2", tags=["large", "accurate"])
    registry.register_model(models["onnx"], "ONNX Model", tags=["small", "optimized"])
    registry.register_model(models["tensorflow"], "TensorFlow Model", tags=["large", "accurate"])

    # List all models
    all_models = registry.list_models()
    assert len(all_models) == 4

    # List by type
    pytorch_models = registry.list_models(model_type=ModelType.PYTORCH)
    assert len(pytorch_models) == 2
    assert pytorch_models[0]["name"] == "PyTorch Model 1"
    assert pytorch_models[1]["name"] == "PyTorch Model 2"

    # List by tag
    small_models = registry.list_models(tag="small")
    assert len(small_models) == 2

    accurate_models = registry.list_models(tag="accurate")
    assert len(accurate_models) == 2

    # Combined filter
    small_pytorch_models = registry.list_models(model_type=ModelType.PYTORCH, tag="small")
    assert len(small_pytorch_models) == 1
    assert small_pytorch_models[0]["name"] == "PyTorch Model 1"


# Test model loading with mocks
class TestModelLoading:

    @patch('model_registry.ModelRegistry._load_pytorch_model')
    def test_load_pytorch_model(self, mock_load, mock_model_files):
        """Test loading a PyTorch model"""
        temp_dir, models = mock_model_files

        # Create mock model and preprocess function
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_load.return_value = (mock_model, mock_preprocess)

        # Create config
        config = {
            "models": {
                "directory": str(Path(temp_dir) / "models")
            }
        }

        # Initialize registry
        registry = ModelRegistry(config)

        # Register a PyTorch model
        model_id = registry.register_model(models["pytorch"], "PyTorch Test")

        # Load the model
        model, preprocess = registry.load_model(model_id)

        # Check that _load_pytorch_model was called
        mock_load.assert_called_once_with(str(models["pytorch"]), 'cpu')

        # Check that the model was loaded
        assert model == mock_model
        assert preprocess == mock_preprocess
        assert model_id in registry.loaded_models

        # Check that loading again uses cached version
        registry.load_model(model_id)
        assert mock_load.call_count == 1  # Still only called once

    @patch('model_registry.ModelRegistry._load_onnx_model')
    def test_load_onnx_model(self, mock_load, mock_model_files):
        """Test loading an ONNX model"""
        temp_dir, models = mock_model_files

        # Create mock model and preprocess function
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_load.return_value = (mock_model, mock_preprocess)

        # Create config
        config = {
            "models": {
                "directory": str(Path(temp_dir) / "models")
            }
        }

        # Initialize registry
        registry = ModelRegistry(config)

        # Register an ONNX model
        model_id = registry.register_model(models["onnx"], "ONNX Test")

        # Load the model
        model, preprocess = registry.load_model(model_id)

        # Check that _load_onnx_model was called
        mock_load.assert_called_once_with(str(models["onnx"]), 'cpu')

        # Check that the model was loaded
        assert model == mock_model
        assert preprocess == mock_preprocess

    def test_unload_model(self, mock_model_files):
        """Test unloading a model"""
        temp_dir, models = mock_model_files

        # Create config
        config = {
            "models": {
                "directory": str(Path(temp_dir) / "models")
            }
        }

        # Initialize registry
        registry = ModelRegistry(config)

        # Register a model
        model_id = registry.register_model(models["pytorch"], "Test Model")

        # Manually add a mock model to loaded_models
        registry.loaded_models[model_id] = (MagicMock(), MagicMock())

        # Unload the model
        result = registry.unload_model(model_id)

        # Check it was unloaded
        assert result is True
        assert model_id not in registry.loaded_models

        # Unload non-existent model
        result = registry.unload_model("nonexistent_id")
        assert result is False

    def test_unload_all_models(self, mock_model_files):
        """Test unloading all models"""
        temp_dir, models = mock_model_files

        # Create config
        config = {
            "models": {
                "directory": str(Path(temp_dir) / "models")
            }
        }

        # Initialize registry
        registry = ModelRegistry(config)

        # Register some models
        model_id1 = registry.register_model(models["pytorch"], "Model 1")
        model_id2 = registry.register_model(models["onnx"], "Model 2")

        # Manually add mock models to loaded_models
        registry.loaded_models[model_id1] = (MagicMock(), MagicMock())
        registry.loaded_models[model_id2] = (MagicMock(), MagicMock())

        # Unload all models
        count = registry.unload_all_models()

        # Check they were unloaded
        assert count == 2
        assert len(registry.loaded_models) == 0


if __name__ == "__main__":
    pytest.main()