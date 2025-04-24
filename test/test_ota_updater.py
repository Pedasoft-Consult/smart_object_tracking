#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the OTA updater.
Uses mocking to test network-dependent functionality.
"""

import sys
import os
import pytest
import json
import time
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import module to test
from updater.ota_updater import OTAUpdater


# Mock config for testing
@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary for testing"""
    return {
        "system": {
            "name": "Smart Object Tracking System",
            "version": "1.0.0"
        },
        "models": {
            "directory": "models",
            "online_model": "yolov5s.pt",
            "offline_model": "yolov5s-fp16.onnx"
        },
        "updates": {
            "enabled": True,
            "check_url": "https://api.example.com/model-updates/check",
            "download_url": "https://storage.example.com/models/",
            "interval": 86400
        }
    }


# Mock for temporary directory
@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model files"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create models directory
        models_dir = Path(tmpdirname) / "models"
        models_dir.mkdir()

        # Create mock model files
        model1 = models_dir / "yolov5s.pt"
        model1.write_bytes(b"mock model data 1")

        model2 = models_dir / "yolov5s-fp16.onnx"
        model2.write_bytes(b"mock model data 2")

        # Create update history file
        history = {
            "last_check": 0,
            "last_update": 0,
            "updates": []
        }
        with open(models_dir / "update_history.json", "w") as f:
            json.dump(history, f)

        yield tmpdirname


# Test OTAUpdater initialization
class TestOTAUpdaterInit:
    def test_initialization(self, mock_config, temp_models_dir):
        """Test that the OTA updater initializes correctly"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        updater = OTAUpdater(config)

        assert updater.check_url == mock_config["updates"]["check_url"]
        assert updater.download_url == mock_config["updates"]["download_url"]
        assert updater.update_interval == mock_config["updates"]["interval"]
        assert updater.models_dir == Path(temp_models_dir) / "models"
        assert updater.history_file == Path(temp_models_dir) / "models" / "update_history.json"

        # Check that update history was loaded
        assert isinstance(updater.update_history, dict)
        assert "last_check" in updater.update_history
        assert "last_update" in updater.update_history
        assert "updates" in updater.update_history


# Test update history functions
class TestUpdateHistory:
    def test_load_history(self, mock_config, temp_models_dir):
        """Test loading update history from file"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Create a custom history file
        history = {
            "last_check": 1234567890,
            "last_update": 1234567000,
            "updates": [
                {
                    "model": "yolov5s.pt",
                    "version": "1.0.1",
                    "timestamp": 1234567000,
                    "date": "2022-01-01T00:00:00",
                    "description": "Test update"
                }
            ]
        }
        with open(Path(temp_models_dir) / "models" / "update_history.json", "w") as f:
            json.dump(history, f)

        # Initialize updater which should load the history
        updater = OTAUpdater(config)

        # Check that history was loaded correctly
        assert updater.update_history["last_check"] == 1234567890
        assert updater.update_history["last_update"] == 1234567000
        assert len(updater.update_history["updates"]) == 1
        assert updater.update_history["updates"][0]["model"] == "yolov5s.pt"
        assert updater.update_history["updates"][0]["version"] == "1.0.1"

    def test_save_history(self, mock_config, temp_models_dir):
        """Test saving update history to file"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Initialize updater
        updater = OTAUpdater(config)

        # Modify history
        updater.update_history["last_check"] = 9876543210
        updater.update_history["updates"].append({
            "model": "new_model.pt",
            "version": "2.0.0",
            "timestamp": 9876543000,
            "date": "2023-01-01T00:00:00",
            "description": "New test update"
        })

        # Save history
        updater._save_update_history()

        # Load history from file and check
        with open(Path(temp_models_dir) / "models" / "update_history.json", "r") as f:
            history = json.load(f)

        assert history["last_check"] == 9876543210
        assert len(history["updates"]) == 1
        assert history["updates"][0]["model"] == "new_model.pt"
        assert history["updates"][0]["version"] == "2.0.0"


# Test file hash calculation
class TestFileHash:
    def test_calculate_file_hash(self, mock_config, temp_models_dir):
        """Test calculating file hash"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Initialize updater
        updater = OTAUpdater(config)

        # Calculate hash of a known file
        model_path = Path(temp_models_dir) / "models" / "yolov5s.pt"

        # Calculate expected hash
        expected_hash = hashlib.sha256(b"mock model data 1").hexdigest()

        # Calculate actual hash
        actual_hash = updater._calculate_file_hash(model_path)

        assert actual_hash == expected_hash


# Test checking for updates
class TestCheckForUpdates:
    @patch('requests.post')
    def test_check_for_updates_success(self, mock_post, mock_config, temp_models_dir):
        """Test successful update check"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "updates": [
                {
                    "filename": "yolov5s.pt",
                    "version": "1.0.1",
                    "download_path": "yolov5s-v1.0.1.pt",
                    "hash": "abcdef1234567890",
                    "hash_algorithm": "sha256",
                    "is_online_model": True
                }
            ]
        }
        mock_post.return_value = mock_response

        # Initialize updater
        updater = OTAUpdater(config)

        # Force check by setting last_check to 0
        updater.update_history["last_check"] = 0

        # Check for updates
        update_info = updater._check_for_updates()

        # Request should not have been made because interval hasn't passed
        mock_post.assert_not_called()

        # No update info should be returned
        assert update_info is None

        # Last check time shouldn't change
        assert updater.update_history["last_check"] == current_time - 1000

    @patch('requests.post')
    def test_check_fails_gracefully(self, mock_post, mock_config, temp_models_dir):
        """Test that update check fails gracefully on network error"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Mock network error
        mock_post.side_effect = Exception("Network error")

        # Initialize updater
        updater = OTAUpdater(config)

        # Force check by setting last_check to 0
        updater.update_history["last_check"] = 0

        # Check for updates
        update_info = updater._check_for_updates()

        # Verify request was attempted
        mock_post.assert_called_once()

        # Check result is None on error
        assert update_info is None

        # Check that last_check was still updated to prevent repeated failures
        assert updater.update_history["last_check"] > 0


# Test downloading and updating models
class TestModelDownload:
    @patch('requests.get')
    def test_download_model_success(self, mock_get, mock_config, temp_models_dir):
        """Test successfully downloading a model"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers.get.return_value = "1000"  # Content length

        # Mock the content as a byte stream
        model_content = b"new model data"
        mock_response.iter_content.return_value = [model_content]
        mock_get.return_value = mock_response

        # Initialize updater
        updater = OTAUpdater(config)

        # Model info
        model_info = {
            "filename": "yolov5s.pt",
            "version": "1.0.1",
            "download_path": "yolov5s-v1.0.1.pt",
            "hash": hashlib.sha256(model_content).hexdigest(),
            "hash_algorithm": "sha256"
        }

        # Download model
        model_path = updater._download_model(model_info)

        # Verify request was made correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == config["updates"]["download_url"] + model_info["download_path"]
        assert kwargs["stream"] is True

        # Check model was downloaded successfully
        assert model_path is not None
        assert model_path.exists()
        assert model_path.name == model_info["filename"]

        # Check content
        with open(model_path, "rb") as f:
            content = f.read()
        assert content == model_content

    @patch('requests.get')
    def test_download_model_hash_failure(self, mock_get, mock_config, temp_models_dir):
        """Test failed hash verification during download"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers.get.return_value = "1000"  # Content length

        # Mock the content as a byte stream
        model_content = b"new model data"
        mock_response.iter_content.return_value = [model_content]
        mock_get.return_value = mock_response

        # Initialize updater
        updater = OTAUpdater(config)

        # Model info with incorrect hash
        model_info = {
            "filename": "yolov5s.pt",
            "version": "1.0.1",
            "download_path": "yolov5s-v1.0.1.pt",
            "hash": "incorrect_hash",
            "hash_algorithm": "sha256"
        }

        # Download model - should fail hash verification
        model_path = updater._download_model(model_info)

        # Verify request was made
        mock_get.assert_called_once()

        # Check that download failed
        assert model_path is None

        # Original model should still exist
        original_model = Path(temp_models_dir) / "models" / "yolov5s.pt"
        assert original_model.exists()

        # Temporary download file should be cleaned up
        temp_file = Path(temp_models_dir) / "models" / "yolov5s.pt.downloading"
        assert not temp_file.exists()

    @patch('requests.get')
    def test_update_model_and_config(self, mock_get, mock_config, temp_models_dir):
        """Test updating a model and configuration"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.headers.get.return_value = "1000"  # Content length

        # Mock the content as a byte stream
        model_content = b"new online model data"
        mock_response.iter_content.return_value = [model_content]
        mock_get.return_value = mock_response

        # Initialize updater
        updater = OTAUpdater(config)

        # Model info for online model
        model_info = {
            "filename": "yolov5m.pt",  # New model name
            "version": "1.0.1",
            "download_path": "yolov5m-v1.0.1.pt",
            "hash": hashlib.sha256(model_content).hexdigest(),
            "hash_algorithm": "sha256",
            "is_online_model": True,  # Flag as online model
            "description": "Updated model with better performance"
        }

        # Before update
        assert config["models"]["online_model"] == "yolov5s.pt"
        assert len(updater.update_history["updates"]) == 0

        # Update model
        result = updater._update_model(model_info)

        # Check result
        assert result is True

        # Check model was downloaded
        new_model_path = Path(temp_models_dir) / "models" / "yolov5m.pt"
        assert new_model_path.exists()

        # Check update history was updated
        assert len(updater.update_history["updates"]) == 1
        assert updater.update_history["updates"][0]["model"] == "yolov5m.pt"
        assert updater.update_history["updates"][0]["version"] == "1.0.1"
        assert updater.update_history["updates"][0]["description"] == "Updated model with better performance"

    @patch('requests.post')
    @patch('requests.get')
    def test_check_and_update_integration(self, mock_get, mock_post, mock_config, temp_models_dir):
        """Test full check_and_update process"""
        # Update config to use temp dir
        config = mock_config.copy()
        config["models"]["directory"] = str(Path(temp_models_dir) / "models")

        # Mock check response
        check_response = MagicMock()
        check_response.status_code = 200
        check_response.json.return_value = {
            "updates": [
                {
                    "filename": "yolov5m.pt",  # New model name
                    "version": "1.0.1",
                    "download_path": "yolov5m-v1.0.1.pt",
                    "hash": "mockhash",
                    "hash_algorithm": "sha256",
                    "is_online_model": True,
                    "description": "Updated model with better performance"
                }
            ]
        }
        mock_post.return_value = check_response

        # Mock download response
        download_response = MagicMock()
        download_response.raise_for_status.return_value = None
        download_response.headers.get.return_value = "1000"  # Content length

        # Mock the content as a byte stream
        model_content = b"new online model data"
        download_response.iter_content.return_value = [model_content]
        mock_get.return_value = download_response

        # Initialize updater with patched _calculate_file_hash method
        updater = OTAUpdater(config)

        # Patch the hash calculation to return the expected hash
        with patch.object(updater, '_calculate_file_hash', return_value="mockhash"):
            # Force check by setting last_check to 0
            updater.update_history["last_check"] = 0

            # Check and update
            result = updater.check_and_update()

            # Check result
            assert result is True

            # Verify check request was made
            mock_post.assert_called_once()

            # Verify download request was made
            mock_get.assert_called_once()

            # Check model was "downloaded"
            new_model_path = Path(temp_models_dir) / "models" / "yolov5m.pt"
            assert new_model_path.exists()

            # Check update history
            assert len(updater.update_history["updates"]) == 1
            assert updater.update_history["updates"][0]["model"] == "yolov5m.pt"


if __name__ == "__main__":
    pytest.main() = updater._check_for_updates()

    # Verify the request was made correctly
    mock_post.assert_called_once()

    # Check call arguments
    args, kwargs = mock_post.call_args
    assert args[0] == config["updates"]["check_url"]
    assert kwargs["headers"] == {'Content-Type': 'application/json'}

    # Parse request data
    request_data = kwargs["json"]
    assert "models" in request_data
    assert "device_info" in request_data
    assert "yolov5s.pt" in request_data["models"]
    assert "yolov5s-fp16.onnx" in request_data["models"]

    # Check the result
    assert update_info == mock_response.json.return_value

    # Check that last_check was updated
    assert updater.update_history["last_check"] > 0


@patch('requests.post')
def test_check_for_updates_no_updates(self, mock_post, mock_config, temp_models_dir):
    """Test checking for updates when none are available"""
    # Update config to use temp dir
    config = mock_config.copy()
    config["models"]["directory"] = str(Path(temp_models_dir) / "models")

    # Mock response with no updates
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "updates": []
    }
    mock_post.return_value = mock_response

    # Initialize updater
    updater = OTAUpdater(config)

    # Force check by setting last_check to 0
    updater.update_history["last_check"] = 0

    # Check for updates
    update_info = updater._check_for_updates()

    # Verify request was made
    mock_post.assert_called_once()

    # Check result
    assert update_info is None


@patch('requests.post')
def test_check_interval_respected(self, mock_post, mock_config, temp_models_dir):
    """Test that update check interval is respected"""
    # Update config to use temp dir
    config = mock_config.copy()
    config["models"]["directory"] = str(Path(temp_models_dir) / "models")

    # Initialize updater
    updater = OTAUpdater(config)

    # Set last_check to recent time
    current_time = time.time()
    updater.update_history["last_check"] = current_time - 1000  # 1000 seconds ago

    # Check for updates
    update_info