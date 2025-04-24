#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API server for Smart Object Tracking System.
Provides remote monitoring and control capabilities.
"""

import os
import sys
import json
import yaml
import time
import logging
import threading
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
import io
import queue

# Add project root to path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Global state variables
tracking_system = None
config = None
latest_frame = None
latest_frame_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=10)  # For streaming
stats = {
    "detections": 0,
    "tracks": 0,
    "fps": 0,
    "start_time": time.time(),
    "is_running": False,
    "is_online": False,
    "last_detection": None
}


def load_config():
    """Load configuration from settings file"""
    config_path = Path(__file__).parent / "configs" / "settings.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        app.logger.error(f"Error loading configuration: {e}")
        return {}


# In the initialize() function of api.py
def initialize():
    """Initialize API server and load dependencies"""
    global config, tracking_system

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration
    config = load_config()

    # Initialize a default offline queue for API responses
    from offline_queue import OfflineQueue  # or from enhanced_offline_queue import EnhancedOfflineQueue
    queue_dir = config.get('offline', {}).get('queue_directory', 'queue')
    tracking_system = type('', (), {})()  # Empty object
    tracking_system.offline_queue = OfflineQueue(queue_dir)  # or EnhancedOfflineQueue

    app.logger.info("API server initialized")


def update_frame(frame, detections=None, tracks=None):
    """
    Update the latest frame with detection and tracking info

    Args:
        frame: Current video frame
        detections: List of detections
        tracks: List of active tracks
    """
    global latest_frame, stats

    if frame is None:
        return

    # Make a copy to avoid modifying the original
    display_frame = frame.copy()

    # Draw detections and tracks if provided
    if detections:
        stats["detections"] = len(detections)
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if tracks:
        stats["tracks"] = len(tracks)
        for track in tracks:
            track_id = track.get('id', 0)
            x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update timestamp
    stats["last_detection"] = time.time()

    # Add FPS info
    fps = stats.get("fps", 0)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update latest frame with lock to avoid corruption
    with latest_frame_lock:
        latest_frame = display_frame

    # Add to streaming queue, dropping frames if full
    try:
        frame_queue.put(display_frame, block=False)
    except queue.Full:
        # Skip frame if queue is full
        pass


def generate_frames():
    """Generator for video streaming frames"""
    while True:
        try:
            # Get frame from queue or wait for new one
            frame = frame_queue.get(timeout=1.0)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()

            # Yield the frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except queue.Empty:
            # No frame available, yield empty frame to keep connection
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n\r\n')
        except Exception as e:
            app.logger.error(f"Error generating video frame: {e}")
            time.sleep(0.1)  # Avoid tight loop on error


# API Routes

@app.route('/api/status')
def get_status():
    """Get current system status"""
    global stats, config

    # Calculate uptime
    uptime = time.time() - stats["start_time"]

    # Get offline queue stats if available
    queue_stats = {}
    if tracking_system and hasattr(tracking_system, 'offline_queue'):
        queue_stats = tracking_system.offline_queue.stats()

    # Build response
    response = {
        "status": "running" if stats["is_running"] else "stopped",
        "mode": "online" if stats["is_online"] else "offline",
        "uptime": uptime,
        "detections": stats["detections"],
        "tracks": stats["tracks"],
        "fps": stats["fps"],
        "last_detection_time": stats["last_detection"],
        "system_info": {
            "name": config.get('system', {}).get('name', "Smart Object Tracking System"),
            "version": config.get('system', {}).get('version', "1.0.0")
        },
        "queue": queue_stats
    }

    return jsonify(response)


@app.route('/api/queue/stats')
def get_queue_stats():
    """Get offline queue statistics"""
    global tracking_system

    try:
        # Check if offline queue is available
        if tracking_system and hasattr(tracking_system, 'offline_queue') and tracking_system.offline_queue:
            stats = tracking_system.offline_queue.stats()
            return jsonify(stats)
        else:
            # Return empty stats instead of 404
            return jsonify({
                "total_items": 0,
                "uploaded_items": 0,
                "pending_items": 0,
                "queue_size_bytes": 0,
                "queue_size_mb": 0,
                "in_memory_items": 0,
                "status": "Queue not initialized yet"
            })

    except Exception as e:
        app.logger.error(f"Error getting queue stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/config')
def get_config():
    """Get current configuration"""
    global config

    # Remove sensitive info if present
    safe_config = config.copy() if config else {}
    if 'updates' in safe_config and 'check_url' in safe_config['updates']:
        safe_config['updates']['check_url'] = "***"
    if 'updates' in safe_config and 'download_url' in safe_config['updates']:
        safe_config['updates']['download_url'] = "***"

    return jsonify(safe_config)


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration parameters"""
    global config

    try:
        new_config = request.json

        # Update config recursively
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        if config is None:
            config = {}

        # Update config
        update_dict(config, new_config)

        # Save to file
        config_path = Path(__file__).parent / "configs" / "settings.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return jsonify({"success": True, "message": "Configuration updated"})

    except Exception as e:
        app.logger.error(f"Error updating configuration: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/start', methods=['POST'])
def start_tracking():
    """Start the tracking system"""
    global tracking_system, stats

    try:
        if stats["is_running"]:
            return jsonify({"success": False, "message": "Tracking is already running"}), 400

        # Get parameters from request
        params = request.json or {}
        source = params.get('source', config.get('input', {}).get('default_source', 0))
        display = params.get('display', False)
        save_video = params.get('save_video', False)

        # Import main module here to avoid circular imports
        from main import start_tracking, setup_logging

        # Create logger
        logger = setup_logging(config)

        # Create args object with parameters
        class Args:
            pass

        args = Args()
        args.source = source
        args.display = display
        args.save_video = save_video
        args.output_dir = params.get('output_dir', "output")
        args.tracker = params.get('tracker', config.get('tracking', {}).get('tracker', 'deep_sort'))

        # Start tracking in a separate thread
        def run_tracking():
            global tracking_system, stats
            try:
                stats["is_running"] = True
                stats["start_time"] = time.time()
                # This function call will block until tracking is complete
                tracking_system = start_tracking(config, logger, args)
                stats["is_running"] = False
            except Exception as e:
                app.logger.error(f"Error in tracking thread: {e}")
                stats["is_running"] = False

        thread = threading.Thread(target=run_tracking, daemon=True)
        thread.start()

        return jsonify({"success": True, "message": "Tracking started"})

    except Exception as e:
        app.logger.error(f"Error starting tracking: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_tracking():
    """Stop the tracking system"""
    global tracking_system, stats

    try:
        if not stats["is_running"]:
            return jsonify({"success": False, "message": "Tracking is not running"}), 400

        # Set flag to indicate stopping
        stats["is_running"] = False

        # TODO: Implement proper shutdown mechanism
        # For now, we'll rely on the tracking loop to detect the flag

        return jsonify({"success": True, "message": "Tracking stop requested"})

    except Exception as e:
        app.logger.error(f"Error stopping tracking: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/frame')
def get_frame():
    """Get the latest frame as JPEG image"""
    global latest_frame

    try:
        with latest_frame_lock:
            if latest_frame is None:
                # Return a blank image if no frame is available
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                _, buffer = cv2.imencode('.jpg', blank)
                response = Response(buffer.tobytes(), mimetype='image/jpeg')
                return response

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', latest_frame)
            response = Response(buffer.tobytes(), mimetype='image/jpeg')
            return response

    except Exception as e:
        app.logger.error(f"Error getting frame: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/stream')
def video_stream():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/queue/sync', methods=['POST'])
def sync_queue():
    """Trigger manual sync of offline queue"""
    global tracking_system

    try:
        if tracking_system and hasattr(tracking_system, 'offline_queue'):
            # Implement sync functionality here
            # This would connect to a server and upload pending items

            # For now, just mark some items as uploaded for testing
            pending = tracking_system.offline_queue.get_pending_items(limit=10)
            if pending:
                item_ids = [item.get('id') for item in pending]
                count = tracking_system.offline_queue.mark_as_uploaded(item_ids)
                return jsonify({"success": True, "synced_items": count})
            else:
                return jsonify({"success": True, "synced_items": 0, "message": "No pending items"})
        else:
            return jsonify({"error": "Offline queue not available"}), 404

    except Exception as e:
        app.logger.error(f"Error syncing queue: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/models')
def list_models():
    """List available detection models"""
    try:
        models_dir = Path(config.get('models', {}).get('directory', 'models'))
        models = []

        if models_dir.exists():
            for model_path in models_dir.glob('*.*'):
                if model_path.is_file() and not model_path.name.startswith('.'):
                    # Get model info
                    stat = model_path.stat()
                    models.append({
                        "name": model_path.name,
                        "path": str(model_path),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "is_online_model": model_path.name == config.get('models', {}).get('online_model'),
                        "is_offline_model": model_path.name == config.get('models', {}).get('offline_model')
                    })

        return jsonify(models)

    except Exception as e:
        app.logger.error(f"Error listing models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/update', methods=['POST'])
def check_model_updates():
    """Check for model updates"""
    try:
        # Import OTAUpdater
        from updater.ota_updater import OTAUpdater

        updater = OTAUpdater(config)
        updated = updater.check_and_update()

        if updated:
            return jsonify({"success": True, "message": "Models updated successfully"})
        else:
            return jsonify({"success": True, "message": "No updates available"})

    except Exception as e:
        app.logger.error(f"Error checking for updates: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/update_history')
def get_update_history():
    """Get model update history"""
    try:
        from updater.ota_updater import OTAUpdater

        updater = OTAUpdater(config)
        history = updater.get_update_history()

        return jsonify(history)

    except Exception as e:
        app.logger.error(f"Error getting update history: {e}")
        return jsonify({"error": str(e)}), 500


# Web interface routes
@app.route('/')
def index():
    """Serve the main web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Object Tracking System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1, h2 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .row { display: flex; flex-wrap: wrap; margin: 0 -15px; }
            .col { flex: 1; padding: 0 15px; min-width: 300px; }
            .panel { background: #f5f5f5; border-radius: 5px; padding: 15px; margin-bottom: 20px; }
            .video-container { width: 100%; }
            video, img { max-width: 100%; border: 1px solid #ddd; }
            button { padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0069d9; }
            button.danger { background: #dc3545; }
            button.danger:hover { background: #c82333; }
            pre { background: #eee; padding: 10px; border-radius: 4px; overflow: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Smart Object Tracking System</h1>

            <div class="row">
                <div class="col">
                    <div class="panel">
                        <h2>Live Feed</h2>
                        <div class="video-container">
                            <img id="video-feed" src="/api/stream" alt="Live video feed">
                        </div>
                    </div>
                </div>

                <div class="col">
                    <div class="panel">
                        <h2>System Status</h2>
                        <div id="status-display">Loading...</div>
                        <div class="controls" style="margin-top: 15px;">
                            <button id="start-btn">Start Tracking</button>
                            <button id="stop-btn" class="danger">Stop Tracking</button>
                        </div>
                    </div>

                    <div class="panel">
                        <h2>Offline Queue</h2>
                        <div id="queue-stats">Loading...</div>
                        <button id="sync-btn" style="margin-top: 10px;">Sync Queue</button>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col">
                    <div class="panel">
                        <h2>Models</h2>
                        <div id="models-list">Loading...</div>
                        <button id="update-btn" style="margin-top: 10px;">Check for Updates</button>
                    </div>
                </div>

                <div class="col">
                    <div class="panel">
                        <h2>Configuration</h2>
                        <pre id="config-display">Loading...</pre>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Refresh status every 2 seconds
            setInterval(() => {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusHtml = `
                            <p><strong>Status:</strong> ${data.status}</p>
                            <p><strong>Mode:</strong> ${data.mode}</p>
                            <p><strong>Uptime:</strong> ${Math.floor(data.uptime / 60)} minutes</p>
                            <p><strong>FPS:</strong> ${data.fps.toFixed(1)}</p>
                            <p><strong>Active Tracks:</strong> ${data.tracks}</p>
                            <p><strong>Detections:</strong> ${data.detections}</p>
                        `;
                        document.getElementById('status-display').innerHTML = statusHtml;
                    })
                    .catch(err => console.error('Error fetching status:', err));

                fetch('/api/queue/stats')
                    .then(response => response.json())
                    .then(data => {
                        const queueHtml = `
                            <p><strong>Pending Items:</strong> ${data.pending_items || 0}</p>
                            <p><strong>Uploaded Items:</strong> ${data.uploaded_items || 0}</p>
                            <p><strong>Queue Size:</strong> ${(data.queue_size_mb || 0).toFixed(2)} MB</p>
                        `;
                        document.getElementById('queue-stats').innerHTML = queueHtml;
                    })
                    .catch(err => {
                        document.getElementById('queue-stats').innerHTML = '<p>Queue not available</p>';
                    });
            }, 2000);

            // Load models list
            fetch('/api/models')
                .then(response => response.json())
                .then(data => {
                    const modelsHtml = data.map(model => `
                        <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #ddd;">
                            <p><strong>${model.name}</strong> ${model.is_online_model ? '(Online)' : ''} ${model.is_offline_model ? '(Offline)' : ''}</p>
                            <p>Size: ${(model.size / (1024*1024)).toFixed(2)} MB</p>
                            <p>Last Modified: ${new Date(model.modified).toLocaleString()}</p>
                        </div>
                    `).join('') || '<p>No models found</p>';
                    document.getElementById('models-list').innerHTML = modelsHtml;
                })
                .catch(err => console.error('Error fetching models:', err));

            // Load configuration
            fetch('/api/config')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('config-display').textContent = JSON.stringify(data, null, 2);
                })
                .catch(err => console.error('Error fetching config:', err));

            // Button event listeners
            document.getElementById('start-btn').addEventListener('click', () => {
                fetch('/api/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ display: false, save_video: false })
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Tracking started');
                })
                .catch(err => console.error('Error starting tracking:', err));
            });

            document.getElementById('stop-btn').addEventListener('click', () => {
                fetch('/api/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Tracking stopped');
                })
                .catch(err => console.error('Error stopping tracking:', err));
            });

            document.getElementById('sync-btn').addEventListener('click', () => {
                fetch('/api/queue/sync', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(`Synced ${data.synced_items} items from queue`);
                })
                .catch(err => console.error('Error syncing queue:', err));
            });

            document.getElementById('update-btn').addEventListener('click', () => {
                fetch('/api/models/update', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || 'Update check complete');
                    // Reload models list
                    fetch('/api/models')
                        .then(response => response.json())
                        .then(data => {
                            const modelsHtml = data.map(model => `
                                <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #ddd;">
                                    <p><strong>${model.name}</strong> ${model.is_online_model ? '(Online)' : ''} ${model.is_offline_model ? '(Offline)' : ''}</p>
                                    <p>Size: ${(model.size / (1024*1024)).toFixed(2)} MB</p>
                                    <p>Last Modified: ${new Date(model.modified).toLocaleString()}</p>
                                </div>
                            `).join('') || '<p>No models found</p>';
                            document.getElementById('models-list').innerHTML = modelsHtml;
                        });
                })
                .catch(err => console.error('Error checking for updates:', err));
            });
        </script>
    </body>
    </html>
    """


# Modify detect_and_track.py to call our update_frame function
def patch_detect_and_track():
    """
    Monkey-patch the detect_and_track module to capture frames and statistics
    """
    import detect_and_track

    # Save the original run function
    original_run = detect_and_track.run

    # Define our patched run function
    def patched_run(*args, **kwargs):
        global stats

        # Check network connectivity
        from utils.connectivity import check_connectivity
        stats["is_online"] = check_connectivity()

        # Call the original function
        return original_run(*args, **kwargs)

    # Replace the original function
    detect_and_track.run = patched_run

    # Patch the code that processes frames
    original_code = """
                if display or save_video:
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    for track in tracks:
                        track_id = track.get('id', 0)
                        x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    """

    # Import the function from this module to avoid circular imports
    from api import update_frame

    # Updated code to also call our update_frame function
    updated_code = """
                # Update API frame
                from api import update_frame
                update_frame(display_frame, detections, tracks)

                if display or save_video:
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        label = f"{det['class_name']} {det['confidence']:.2f}"
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    for track in tracks:
                        track_id = track.get('id', 0)
                        x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    """

    # This is pseudo-code since we can't actually patch the source code at runtime
    # In a real implementation, we'd need to modify the file or use a more sophisticated
    # approach to intercept the frames

    app.logger.info("Patched detect_and_track module")


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(description="API Server for Smart Object Tracking System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server to")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    # Initialize API
    initialize()

    # Patch detect_and_track module (in a real implementation)
    # patch_detect_and_track()

    # Start Flask server
    app.run(host=args.host, port=args.port, debug=args.debug)