#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OTA (Over-The-Air) updater for models.
Checks for updated models and downloads them automatically.
"""

import os
import sys
import json
import yaml
import time
import logging
import hashlib
import requests
import shutil
from pathlib import Path
from datetime import datetime


class OTAUpdater:
    """Handles OTA updates for models"""

    def __init__(self, config):
        """
        Initialize OTA updater

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger('OTAUpdater')

        # Get update settings
        update_config = config.get('updates', {})
        self.check_url = update_config.get('check_url', '')
        self.download_url = update_config.get('download_url', '')
        self.update_interval = update_config.get('interval', 86400)  # Default: 1 day
        self.models_dir = Path(config.get('models', {}).get('directory', 'models'))

        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Set update history file
        self.history_file = self.models_dir / 'update_history.json'
        self.update_history = self._load_update_history()

    def _load_update_history(self):
        """
        Load update history from file

        Returns:
            Update history dictionary
        """
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading update history: {e}")

        # Create default history
        return {
            "last_check": 0,
            "last_update": 0,
            "updates": []
        }

    def _save_update_history(self):
        """Save update history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.update_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving update history: {e}")

    def _calculate_file_hash(self, file_path, algorithm='sha256'):
        """
        Calculate hash of file

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            Hash string
        """
        hasher = getattr(hashlib, algorithm)()

        with open(file_path, 'rb') as f:
            # Read and hash file in chunks
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _check_for_updates(self):
        """
        Check for model updates

        Returns:
            None if no updates, or dictionary with update info
        """
        if not self.check_url:
            self.logger.warning("Update check URL not configured")
            return None

        try:
            # Check if it's time to check for updates
            current_time = time.time()
            if (current_time - self.update_history["last_check"]) < self.update_interval:
                self.logger.debug("Update check interval not reached")
                return None

            # Update last check time
            self.update_history["last_check"] = current_time
            self._save_update_history()

            # Get current model files
            model_files = {}
            for model_path in self.models_dir.glob('*.*'):
                if model_path.is_file() and not model_path.name.startswith('.'):
                    model_files[model_path.name] = {
                        "path": str(model_path),
                        "size": model_path.stat().st_size,
                        "modified": model_path.stat().st_mtime
                    }

            # Create request data
            request_data = {
                "models": list(model_files.keys()),
                "device_info": {
                    "platform": sys.platform,
                    "version": self.config.get('system', {}).get('version', '1.0')
                }
            }

            # Send request to check for updates
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.check_url,
                json=request_data,
                headers=headers,
                timeout=30
            )

            # Check response
            if response.status_code != 200:
                self.logger.error(f"Update check failed: {response.status_code} {response.reason}")
                return None

            # Parse response
            update_info = response.json()

            if not update_info.get('updates', []):
                self.logger.info("No updates available")
                return None

            return update_info

        except requests.RequestException as e:
            self.logger.error(f"Update check request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error checking for updates: {e}")
            return None

    def _download_model(self, model_info):
        """
        Download model file

        Args:
            model_info: Model info dictionary

        Returns:
            Path to downloaded file or None if failed
        """
        if not self.download_url:
            self.logger.warning("Download URL not configured")
            return None

        model_name = model_info['filename']
        model_version = model_info.get('version', 'unknown')
        model_url = f"{self.download_url}{model_info.get('download_path', model_name)}"

        try:
            self.logger.info(f"Downloading model {model_name} (version {model_version})")

            # Download to temporary file
            temp_file = self.models_dir / f"{model_name}.downloading"

            # Stream download to handle large files
            with requests.get(model_url, stream=True, timeout=300) as response:
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress for large files
                            if total_size > 10 * 1024 * 1024 and downloaded % (1024 * 1024) == 0:  # Log every MB
                                percent = 100 * downloaded / total_size if total_size > 0 else 0
                                self.logger.info(
                                    f"Download progress: {downloaded / 1024 / 1024:.1f} MB / {total_size / 1024 / 1024:.1f} MB ({percent:.1f}%)")

            # Verify hash if provided
            if 'hash' in model_info and 'hash_algorithm' in model_info:
                file_hash = self._calculate_file_hash(temp_file, model_info['hash_algorithm'])
                if file_hash != model_info['hash']:
                    self.logger.error(f"Hash verification failed for {model_name}")
                    os.unlink(temp_file)
                    return None

            # Backup existing model if it exists
            model_path = self.models_dir / model_name
            if model_path.exists():
                backup_name = f"{model_name}.backup.{int(time.time())}"
                backup_path = self.models_dir / backup_name
                shutil.move(model_path, backup_path)
                self.logger.info(f"Backed up existing model to {backup_name}")

            # Move temporary file to final location
            shutil.move(temp_file, model_path)

            self.logger.info(f"Model {model_name} downloaded successfully")
            return model_path

        except requests.RequestException as e:
            self.logger.error(f"Model download request failed: {e}")
            if temp_file.exists():
                os.unlink(temp_file)
            return None
        except Exception as e:
            self.logger.error(f"Error downloading model: {e}")
            if temp_file.exists():
                os.unlink(temp_file)
            return None

    def _update_model(self, model_info):
        """
        Update model

        Args:
            model_info: Model info dictionary

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Download model
            model_path = self._download_model(model_info)
            if not model_path:
                return False

            # Update configuration if needed
            models_config = self.config.get('models', {})
            model_name = model_info['filename']

            if model_info.get('is_online_model', False):
                models_config['online_model'] = model_name
            if model_info.get('is_offline_model', False):
                models_config['offline_model'] = model_name

            # Record update in history
            update_record = {
                "model": model_name,
                "version": model_info.get('version', 'unknown'),
                "timestamp": time.time(),
                "date": datetime.now().isoformat(),
                "description": model_info.get('description', '')
            }

            self.update_history["last_update"] = time.time()
            self.update_history["updates"].append(update_record)
            self._save_update_history()

            self.logger.info(f"Model {model_name} updated to version {model_info.get('version', 'unknown')}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False

    def check_and_update(self):
        """
        Check for updates and apply them

        Returns:
            True if updates were installed, False otherwise
        """
        try:
            # Check for updates
            update_info = self._check_for_updates()
            if not update_info:
                return False

            # Process updates
            updated = False
            for model_info in update_info.get('updates', []):
                if self._update_model(model_info):
                    updated = True

            return updated

        except Exception as e:
            self.logger.error(f"Error in check_and_update: {e}")
            return False

    def get_update_history(self):
        """
        Get update history

        Returns:
            Update history dictionary
        """
        return self.update_history


# Standalone test function
def test_ota_updater():
    """Test OTA updater functionality"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('OTAUpdaterTest')

    # Load config
    config_path = Path(__file__).parents[1] / "configs" / "settings.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return

    # Create updater
    updater = OTAUpdater(config)

    # Check and update
    if updater.check_and_update():
        logger.info("Updates installed successfully")
    else:
        logger.info("No updates installed")

    # Show update history
    history = updater.get_update_history()
    logger.info(f"Last check: {datetime.fromtimestamp(history['last_check']).isoformat()}")
    logger.info(
        f"Last update: {datetime.fromtimestamp(history['last_update']).isoformat() if history['last_update'] > 0 else 'Never'}")
    logger.info(f"Update count: {len(history['updates'])}")

    for update in history['updates']:
        logger.info(f"  - {update['date']}: {update['model']} v{update['version']}")


if __name__ == "__main__":
    test_ota_updater()