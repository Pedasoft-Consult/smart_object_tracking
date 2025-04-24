#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Client for Smart Object Tracking System.
Provides a Python interface to interact with the tracking system API.
"""

import requests
import json
import time
import base64
import cv2
import numpy as np
import io
from pathlib import Path
import logging
import threading


class TrackingAPIClient:
    """Client for interacting with Smart Object Tracking System API"""

    def __init__(self, base_url="http://localhost:5000", api_key=None):
        """
        Initialize API client

        Args:
            base_url: Base URL of the API server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.logger = logging.getLogger('TrackingAPIClient')

        # Set up default headers
        self.headers = {
            'Content-Type': 'application/json'
        }

        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'

    def _get(self, endpoint, params=None):
        """
        Make a GET request to the API

        Args:
            endpoint: API endpoint
            params: Optional query parameters

        Returns:
            Response data or None if error
        """
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Error making API request: {e}")
            return None

    def _post(self, endpoint, data=None):
        """
        Make a POST request to the API

        Args:
            endpoint: API endpoint
            data: Optional data to send

        Returns:
            Response data or None if error
        """
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.post(url, headers=self.headers, json=data)

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"Error making API request: {e}")
            return None

    def get_status(self):
        """
        Get current system status

        Returns:
            Status data or None if error
        """
        return self._get('/api/status')

    def get_config(self):
        """
        Get current configuration

        Returns:
            Configuration data or None if error
        """
        return self._get('/api/config')

    def update_config(self, config_changes):
        """
        Update configuration parameters

        Args:
            config_changes: Dictionary of configuration changes

        Returns:
            Response data or None if error
        """
        return self._post('/api/config', config_changes)

    def start_tracking(self, source=None, display=False, save_video=False, output_dir=None, tracker=None):
        """
        Start the tracking system

        Args:
            source: Video source (camera index or file path)
            display: Whether to display output
            save_video: Whether to save output video
            output_dir: Output directory for saved videos
            tracker: Tracker type to use

        Returns:
            Response data or None if error
        """
        params = {
            'source': source,
            'display': display,
            'save_video': save_video
        }

        if output_dir:
            params['output_dir'] = output_dir

        if tracker:
            params['tracker'] = tracker

        return self._post('/api/start', params)

    def stop_tracking(self):
        """
        Stop the tracking system

        Returns:
            Response data or None if error
        """
        return self._post('/api/stop')

    def get_frame(self, as_numpy=True):
        """
        Get the latest frame

        Args:
            as_numpy: Whether to return as numpy array (otherwise bytes)

        Returns:
            Frame data or None if error
        """
        try:
            url = f"{self.base_url}/api/frame"
            response = requests.get(url)

            if response.status_code == 200:
                if as_numpy:
                    # Convert to numpy array
                    img_array = np.frombuffer(response.content, np.uint8)
                    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                else:
                    return response.content
            else:
                self.logger.error(f"API error: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None

    def save_frame(self, filename):
        """
        Save the latest frame to file

        Args:
            filename: Path to save frame to

        Returns:
            True if successful, False otherwise
        """
        frame = self.get_frame(as_numpy=True)

        if frame is not None:
            try:
                cv2.imwrite(filename, frame)
                return True
            except Exception as e:
                self.logger.error(f"Error saving frame: {e}")
                return False

        return False

    def get_queue_stats(self):
        """
        Get offline queue statistics

        Returns:
            Queue statistics or None if error
        """
        return self._get('/api/queue/stats')

    def sync_queue(self):
        """
        Trigger manual sync of offline queue

        Returns:
            Response data or None if error
        """
        return self._post('/api/queue/sync')

    def list_models(self):
        """
        List available detection models

        Returns:
            List of models or None if error
        """
        return self._get('/api/models')

    def check_model_updates(self):
        """
        Check for model updates

        Returns:
            Response data or None if error
        """
        return self._post('/api/models/update')

    def get_update_history(self):
        """
        Get model update history

        Returns:
            Update history or None if error
        """
        return self._get('/api/update_history')

    def stream_frames(self, callback, interval=0.1, stop_event=None):
        """
        Stream frames continuously in a background thread

        Args:
            callback: Function to call with each frame
            interval: Time between frames in seconds
            stop_event: Optional threading event to stop streaming

        Returns:
            Thread object
        """
        if stop_event is None:
            stop_event = threading.Event()

        def streaming_thread():
            while not stop_event.is_set():
                try:
                    frame = self.get_frame()
                    if frame is not None:
                        callback(frame)
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Error in streaming thread: {e}")
                    time.sleep(1)  # Wait longer on error

        thread = threading.Thread(target=streaming_thread)
        thread.daemon = True
        thread.start()

        return thread


class TrackingAPIMonitor:
    """Monitor for tracking results from the Smart Object Tracking System API"""

    def __init__(self, client):
        """
        Initialize monitor

        Args:
            client: TrackingAPIClient instance
        """
        self.client = client
        self.logger = logging.getLogger('TrackingAPIMonitor')
        self.stop_event = threading.Event()
        self.status_thread = None
        self.frame_thread = None
        self.status_callbacks = []
        self.frame_callbacks = []
        self.latest_status = None
        self.latest_frame = None

    def add_status_callback(self, callback):
        """
        Add a callback for status updates

        Args:
            callback: Function to call with status data
        """
        self.status_callbacks.append(callback)

    def add_frame_callback(self, callback):
        """
        Add a callback for frame updates

        Args:
            callback: Function to call with frame data
        """
        self.frame_callbacks.append(callback)

    def _status_thread_func(self):
        """Status monitoring thread function"""
        while not self.stop_event.is_set():
            try:
                status = self.client.get_status()
                if status:
                    self.latest_status = status
                    for callback in self.status_callbacks:
                        try:
                            callback(status)
                        except Exception as e:
                            self.logger.error(f"Error in status callback: {e}")

                time.sleep(1)  # Check status every second

            except Exception as e:
                self.logger.error(f"Error in status thread: {e}")
                time.sleep(5)  # Wait longer on error

    def _frame_thread_func(self):
        """Frame monitoring thread function"""
        while not self.stop_event.is_set():
            try:
                frame = self.client.get_frame()
                if frame is not None:
                    self.latest_frame = frame
                    for callback in self.frame_callbacks:
                        try:
                            callback(frame)
                        except Exception as e:
                            self.logger.error(f"Error in frame callback: {e}")

                time.sleep(0.1)  # 10 FPS max

            except Exception as e:
                self.logger.error(f"Error in frame thread: {e}")
                time.sleep(1)  # Wait longer on error

    def start(self):
        """
        Start monitoring

        Returns:
            Self for chaining
        """
        if self.status_thread is None or not self.status_thread.is_alive():
            self.stop_event.clear()
            self.status_thread = threading.Thread(target=self._status_thread_func)
            self.status_thread.daemon = True
            self.status_thread.start()

        if self.frame_thread is None or not self.frame_thread.is_alive():
            self.frame_thread = threading.Thread(target=self._frame_thread_func)
            self.frame_thread.daemon = True
            self.frame_thread.start()

        return self

    def stop(self):
        """
        Stop monitoring

        Returns:
            Self for chaining
        """
        self.stop_event.set()

        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=1.0)

        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1.0)

        return self


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create client
    client = TrackingAPIClient("http://localhost:5000")

    # Get and print system status
    status = client.get_status()
    print(f"System status: {status}")


    # Start monitoring
    def on_status_update(status):
        print(f"FPS: {status.get('fps', 0):.1f}, Tracks: {status.get('tracks', 0)}")


    def on_frame_update(frame):
        # Display frame
        cv2.imshow("Tracking Monitor", frame)
        cv2.waitKey(1)


    # Create monitor
    monitor = TrackingAPIMonitor(client)
    monitor.add_status_callback(on_status_update)
    monitor.add_frame_callback(on_frame_update)

    # Start monitoring
    monitor.start()

    # Start tracking
    client.start_tracking(source=0)  # Use default camera

    try:
        # Run for 30 seconds
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop tracking
        client.stop_tracking()

        # Stop monitoring
        monitor.stop()

        # Close any open windows
        cv2.destroyAllWindows()