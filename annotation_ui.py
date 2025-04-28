#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Annotation Interface for Smart Object Tracking System.
Provides a web interface for manual annotation and correction of detections.
"""

import os
import json
import time
import logging
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template_string, Response

# Create blueprint for annotation routes
annotation_bp = Blueprint('annotation', __name__)

# Initialize logger
logger = logging.getLogger('AnnotationUI')

# Global feedback manager reference
feedback_manager = None


def init_annotation_ui(app, feedback_mgr):
    """
    Initialize the annotation UI.

    Args:
        app: Flask application instance
        feedback_mgr: Feedback manager instance
    """
    global feedback_manager
    feedback_manager = feedback_mgr

    # Register blueprint with the app
    app.register_blueprint(annotation_bp, url_prefix='/api/annotation')

    logger.info("Annotation UI initialized")


@annotation_bp.route('/')
def annotation_interface():
    """Serve the annotation interface HTML"""
    return render_template_string(ANNOTATION_HTML)


@annotation_bp.route('/frame')
def get_frame():
    """Get the latest frame for annotation"""
    try:
        # Import get_frame function from API module
        from api import latest_frame, latest_frame_lock, get_frame as api_get_frame

        # Use the existing API function
        return api_get_frame()
    except ImportError:
        # Fallback if API module not available
        logger.error("Failed to import frame from API module")

        # Return a blank image if no frame is available
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', blank)
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        return response


@annotation_bp.route('/submit', methods=['POST'])
def submit_annotation():
    """Submit annotation from web interface"""
    if feedback_manager is None:
        return jsonify({"success": False, "error": "Feedback manager not initialized"}), 500

    try:
        # Get JSON data from request
        data = request.json

        # Validate required fields
        if 'frame_id' not in data:
            return jsonify({"success": False, "error": "Frame ID is required"}), 400

        if 'original_detections' not in data:
            return jsonify({"success": False, "error": "Original detections are required"}), 400

        if 'corrected_detections' not in data:
            return jsonify({"success": False, "error": "Corrected detections are required"}), 400

        # Check if image is included
        image = None
        if 'image_data' in data and data['image_data']:
            # Decode base64 image
            try:
                img_data = base64.b64decode(data['image_data'].split(',')[1])
                img_array = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Error decoding image: {e}")

        # Add feedback to manager
        feedback_id = feedback_manager.add_feedback(
            frame_id=data['frame_id'],
            original_detections=data['original_detections'],
            corrected_detections=data['corrected_detections'],
            image=image,
            metadata=data.get('metadata')
        )

        if feedback_id:
            logger.info(f"Feedback submitted: {feedback_id}")
            return jsonify({
                "success": True,
                "feedback_id": feedback_id,
                "message": "Annotation submitted successfully"
            })
        else:
            logger.error("Failed to add feedback")
            return jsonify({"success": False, "error": "Failed to add feedback"}), 500

    except Exception as e:
        logger.error(f"Error submitting annotation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

@annotation_bp.route('/detections')
def get_detections():
    try:
        import api
        stats = getattr(api, "stats", None)

        if stats is None:
            return jsonify({"success": False, "error": "Stats not initialized"}), 500

        return jsonify({
            "success": True,
            "detections": stats.get("detections", []),
            "tracks": stats.get("tracks", [])
        })
    except Exception as e:
        logger.error(f"Failed to get detections: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@annotation_bp.route('/statistics')
def get_statistics():
    """Get feedback statistics"""
    if feedback_manager is None:
        return jsonify({"success": False, "error": "Feedback manager not initialized"}), 500

    try:
        # Get statistics from feedback manager
        stats = feedback_manager.get_statistics()
        return jsonify({"success": True, "statistics": stats})

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@annotation_bp.route('/history')
def get_history():
    """Get feedback history"""
    if feedback_manager is None:
        return jsonify({"success": False, "error": "Feedback manager not initialized"}), 500

    try:
        # Get parameters
        limit = request.args.get('limit', 10, type=int)
        offset = request.args.get('offset', 0, type=int)
        processed = request.args.get('processed', None)

        if processed is not None:
            processed = processed.lower() == 'true'

        # Get feedback items
        items = feedback_manager.list_feedback_items(limit=limit, offset=offset, processed=processed)

        # Return items without large image data
        sanitized_items = []
        for item in items:
            # Create a copy without image data
            sanitized = item.copy()
            if 'image_path' in sanitized:
                sanitized['has_image'] = True
                # Remove actual path for security
                sanitized['image_path'] = os.path.basename(sanitized['image_path'])
            else:
                sanitized['has_image'] = False

            sanitized_items.append(sanitized)

        return jsonify({
            "success": True,
            "items": sanitized_items,
            "total": feedback_manager.get_statistics().get("total_items", 0)
        })

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@annotation_bp.route('/classes')
def get_classes():
    """Get available class names"""
    try:
        # Try to get classes from the model
        from api import load_config
        config = load_config()

        # Try to find class names from multiple sources
        class_names = []

        # Check dataset directory for classes.txt
        dataset_dir = config.get('dataset', {}).get('directory', 'dataset')
        classes_file = os.path.join(dataset_dir, 'classes.txt')

        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]

        # If no classes found, use COCO classes as default
        if not class_names:
            class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]

        return jsonify({
            "success": True,
            "classes": [{"id": i, "name": name} for i, name in enumerate(class_names)]
        })

    except Exception as e:
        logger.error(f"Error getting classes: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# HTML template for the annotation interface
ANNOTATION_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detection Annotation Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f7fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #2c3e50;
        }
        .toolbar {
            background-color: #34495e;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .controls {
            display: flex;
            gap: 10px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #2980b9;
        }
        button.delete {
            background-color: #e74c3c;
        }
        button.delete:hover {
            background-color: #c0392b;
        }
        button.success {
            background-color: #2ecc71;
        }
        button.success:hover {
            background-color: #27ae60;
        }
        select, input {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #bdc3c7;
        }
        .canvas-container {
            position: relative;
            border: 2px solid #34495e;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 20px;
            background-color: #000;
            width: 100%;
            height: 600px;
        }
        #annotationCanvas {
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
            z-index: 10;
        }
        #videoFeed {
            position: absolute;
            top: 0;
            left: 0;
            max-width: 100%;
            max-height: 100%;
            z-index: 5;
        }
        .data-panel {
            display: flex;
            gap: 20px;
        }
        .box-list {
            flex: 1;
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            height: 200px;
            overflow-y: auto;
        }
        .box-item {
            padding: 8px;
            margin-bottom: 5px;
            border-radius: 4px;
            background-color: #f1f1f1;
            cursor: pointer;
        }
        .box-item.selected {
            background-color: #3498db;
            color: white;
        }
        .feedback {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 15px;
            border-radius: 5px;
            color: white;
            display: none;
            max-width: 300px;
        }
        .feedback.success {
            background-color: #2ecc71;
        }
        .feedback.error {
            background-color: #e74c3c;
        }
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        #stats {
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            display: none;
        }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Detection Annotation Tool</h1>

        <div class="toolbar">
            <div class="controls">
                <button id="pauseBtn">Pause</button>
                <button id="refreshBtn">Refresh Frame</button>
                <select id="classSelect">
                    <option value="0">Loading classes...</option>
                </select>
            </div>
            <div class="controls">
                <button id="addBoxBtn">Add Box</button>
                <button id="editBoxBtn">Edit Selected</button>
                <button id="deleteBoxBtn" class="delete">Delete Selected</button>
            </div>
        </div>

        <div class="canvas-container">
            <img id="videoFeed" src="/api/annotation/frame" alt="Video Feed">
            <canvas id="annotationCanvas"></canvas>
        </div>

        <div class="data-panel">
            <div class="box-list">
                <h2>Original Detections</h2>
                <div id="originalBoxes"></div>
            </div>
            <div class="box-list">
                <h2>Corrected Detections</h2>
                <div id="correctedBoxes"></div>
            </div>
        </div>

        <div class="button-group">
            <button id="resetBtn" class="delete">Reset Corrections</button>
            <button id="submitBtn" class="success">Submit Corrections</button>
        </div>

        <div id="stats">
            <h2>Feedback Statistics</h2>
            <div id="statsContent">Loading...</div>
        </div>
    </div>

    <div class="feedback" id="feedbackMessage"></div>

    <div class="loading" id="loadingOverlay">
        <div class="spinner"></div>
    </div>

    <script>
        // Global variables
        let canvas = document.getElementById('annotationCanvas');
        let ctx = canvas.getContext('2d');
        let videoFeed = document.getElementById('videoFeed');
        let isPaused = false;
        let isDrawing = false;
        let startX, startY;
        let currentBox = null;
        let selectedBox = null;
        let isAddingBox = false;
        let isEditingBox = false;
        let originalDetections = [];
        let correctedDetections = [];
        let classes = [];
        let frameId = Date.now().toString();
        let refreshInterval;

        // Initialize
        function init() {
            // Set canvas size to match video feed
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);

            // Load classes
            loadClasses();

            // Load initial detections
            loadDetections();

            // Start refresh interval
            startRefresh();

            // Set up event listeners
            document.getElementById('pauseBtn').addEventListener('click', togglePause);
            document.getElementById('refreshBtn').addEventListener('click', refreshFrame);
            document.getElementById('addBoxBtn').addEventListener('click', startAddBox);
            document.getElementById('editBoxBtn').addEventListener('click', startEditBox);
            document.getElementById('deleteBoxBtn').addEventListener('click', deleteSelectedBox);
            document.getElementById('resetBtn').addEventListener('click', resetCorrections);
            document.getElementById('submitBtn').addEventListener('click', submitCorrections);

            // Canvas event listeners
            canvas.addEventListener('mousedown', handleMouseDown);
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseup', handleMouseUp);

            // Load statistics
            loadStatistics();
        }

        function resizeCanvas() {
            canvas.width = videoFeed.clientWidth;
            canvas.height = videoFeed.clientHeight;
            drawBoxes();
        }

        function startRefresh() {
            if (!refreshInterval) {
                refreshInterval = setInterval(() => {
                    if (!isPaused) {
                        refreshFrame();
                    }
                }, 1000); // Refresh every second
            }
        }

        function stopRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
            }
        }

        function togglePause() {
            isPaused = !isPaused;
            document.getElementById('pauseBtn').textContent = isPaused ? 'Resume' : 'Pause';
            if (!isPaused) {
                refreshFrame();
            }
        }

        function refreshFrame() {
            videoFeed.src = `/api/annotation/frame?t=${Date.now()}`;
            loadDetections();
            frameId = Date.now().toString();
        }

        function loadClasses() {
            fetch('/api/annotation/classes')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        classes = data.classes;
                        let select = document.getElementById('classSelect');
                        select.innerHTML = '';

                        classes.forEach(classObj => {
                            let option = document.createElement('option');
                            option.value = classObj.id;
                            option.textContent = classObj.name;
                            select.appendChild(option);
                        });
                    }
                })
                .catch(err => {
                    showFeedback('Error loading classes: ' + err.message, false);
                });
        }

        function loadDetections() {
            fetch('/api/annotation/detections')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Convert API format to our format
if (Array.isArray(data.detections)) {
    originalDetections = data.detections.map(det => {
        return {
            bbox: det.bbox,
            class_id: det.class_id,
            confidence: det.confidence,
            class_name: det.class_name || getClassName(det.class_id)
        };
    });
} else {
    originalDetections = [];
    showFeedback('Warning: detections not returned as array', false);
}


                        // Initialize corrected detections if empty
                        if (correctedDetections.length === 0) {
                            correctedDetections = JSON.parse(JSON.stringify(originalDetections));
                        }

                        updateBoxLists();
                        drawBoxes();
                    }
                })
                .catch(err => {
                    showFeedback('Error loading detections: ' + err.message, false);
                });
        }

        function loadStatistics() {
            fetch('/api/annotation/statistics')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        const stats = data.statistics;
                        let content = '';

                        content += createStatItem('Total feedback items', stats.total_items || 0);
                        content += createStatItem('Processed items', stats.processed_items || 0);
                        content += createStatItem('Pending items', stats.pending_items || 0);
                        content += createStatItem('Added to dataset', stats.added_to_dataset || 0);

                        if (stats.correction_stats) {
                            content += '<h3>Correction Statistics</h3>';
                            content += createStatItem('Total corrections', stats.correction_stats.total_corrections || 0);
                            content += createStatItem('Objects added', stats.correction_stats.added_objects || 0);
                            content += createStatItem('Objects removed', stats.correction_stats.removed_objects || 0);
                            content += createStatItem('Objects modified', stats.correction_stats.modified_objects || 0);
                        }

                        document.getElementById('statsContent').innerHTML = content;
                    }
                })
                .catch(err => {
                    document.getElementById('statsContent').textContent = 'Error loading statistics';
                });
        }

        function createStatItem(label, value) {
            return `<div class="stat-item"><div>${label}:</div><div>${value}</div></div>`;
        }

        function updateBoxLists() {
            let originalList = document.getElementById('originalBoxes');
            let correctedList = document.getElementById('correctedBoxes');

            originalList.innerHTML = '';
            correctedList.innerHTML = '';

            originalDetections.forEach((box, index) => {
                let div = document.createElement('div');
                div.className = 'box-item';
                div.textContent = `${box.class_name} (${(box.confidence * 100).toFixed(1)}%)`;
                div.title = `Box ${index + 1}: [${box.bbox.map(Math.round).join(', ')}]`;
                originalList.appendChild(div);
            });

            correctedDetections.forEach((box, index) => {
                let div = document.createElement('div');
                div.className = 'box-item';
                div.dataset.index = index;
                div.textContent = `${box.class_name} (${(box.confidence * 100).toFixed(1)}%)`;
                div.title = `Box ${index + 1}: [${box.bbox.map(Math.round).join(', ')}]`;

                if (selectedBox !== null && selectedBox === index) {
                    div.classList.add('selected');
                }

                div.addEventListener('click', () => {
                    selectBox(index);
                });

                correctedList.appendChild(div);
            });
        }

        function selectBox(index) {
            selectedBox = (selectedBox === index) ? null : index;
            updateBoxLists();
            drawBoxes();
        }

        function startAddBox() {
            isAddingBox = true;
            isEditingBox = false;
            selectedBox = null;
            currentBox = null;
            updateBoxLists();
            drawBoxes();
            showFeedback('Click and drag to add a new box', true);
        }

        function startEditBox() {
            if (selectedBox === null) {
                showFeedback('Please select a box to edit first', false);
                return;
            }

            isEditingBox = true;
            isAddingBox = false;
            showFeedback('Click and drag to edit the selected box', true);
        }

        function deleteSelectedBox() {
            if (selectedBox === null) {
                showFeedback('Please select a box to delete first', false);
                return;
            }

            correctedDetections.splice(selectedBox, 1);
            selectedBox = null;
            updateBoxLists();
            drawBoxes();
            showFeedback('Box deleted', true);
        }

        function resetCorrections() {
            correctedDetections = JSON.parse(JSON.stringify(originalDetections));
            selectedBox = null;
            updateBoxLists();
            drawBoxes();
            showFeedback('Corrections reset to original detections', true);
        }

        function submitCorrections() {
            // Show loading overlay
            document.getElementById('loadingOverlay').style.display = 'flex';

            // Get base64 representation of the current frame
            let canvas = document.createElement('canvas');
            canvas.width = videoFeed.naturalWidth;
            canvas.height = videoFeed.naturalHeight;
            let ctx = canvas.getContext('2d');
            ctx.drawImage(videoFeed, 0, 0);
            let imageData = canvas.toDataURL('image/jpeg');

            // Prepare data
            let data = {
                frame_id: frameId,
                original_detections: originalDetections,
                corrected_detections: correctedDetections,
                image_data: imageData,
                metadata: {
                    timestamp: Date.now(),
                    source: 'annotation_ui'
                }
            };

            // Submit data
            fetch('/api/annotation/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading overlay
                document.getElementById('loadingOverlay').style.display = 'none';

                if (data.success) {
                    showFeedback('Corrections submitted successfully! Feedback ID: ' + data.feedback_id, true);
                    // Reset and get new frame
                    correctedDetections = [];
                    selectedBox = null;
                    refreshFrame();
                    loadStatistics();
                } else {
                    showFeedback('Error: ' + (data.error || 'Unknown error'), false);
                }
            })
            .catch(err => {
                // Hide loading overlay
                document.getElementById('loadingOverlay').style.display = 'none';
                showFeedback('Error: ' + err.message, false);
            });
        }

        function handleMouseDown(e) {
            if (!isPaused) {
                togglePause(); // Pause when starting to draw
            }

            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;

            if (isAddingBox) {
                isDrawing = true;
                currentBox = { x: startX, y: startY, width: 0, height: 0 };
            } else if (isEditingBox && selectedBox !== null) {
                // Check if click is on the selected box
                const box = correctedDetections[selectedBox];
                const [x1, y1, x2, y2] = translateBox(box.bbox);

                if (startX >= x1 && startX <= x2 && startY >= y1 && startY <= y2) {
                    isDrawing = true;
                    currentBox = {
                        x: x1,
                        y: y1,
                        width: x2 - x1,
                        height: y2 - y1,
                        offsetX: startX - x1,
                        offsetY: startY - y1
                    };
                }
            }
        }

        function handleMouseMove(e) {
            if (!isDrawing) return;

            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            if (isAddingBox) {
                currentBox.width = mouseX - startX;
                currentBox.height = mouseY - startY;
            } else if (isEditingBox && selectedBox !== null) {
                // Move the box
                currentBox.x = mouseX - currentBox.offsetX;
                currentBox.y = mouseY - currentBox.offsetY;
            }

            drawBoxes();
        }

        function handleMouseUp(e) {
            if (!isDrawing) return;
            isDrawing = false;

            if (isAddingBox) {
                // Normalize box (handle negative width/height)
                let x = currentBox.width < 0 ? startX + currentBox.width : startX;
                let y = currentBox.height < 0 ? startY + currentBox.height : startY;
                let width = Math.abs(currentBox.width);
                let height = Math.abs(currentBox.height);

                // Only add if box has some size
                if (width > 5 && height > 5) {
                    // Get selected class
                    let classId = parseInt(document.getElementById('classSelect').value);
                    let className = getClassName(classId);

                    // Convert to API format
                    let bbox = [
                        x, y, x + width, y + height
                    ];

                    // Scale to original image dimensions
                    bbox = scaleBoxToOriginal(bbox);

                    // Add to corrected detections
                    correctedDetections.push({
                        bbox: bbox,
                        class_id: classId,
                        class_name: className,
                        confidence: 1.0 // User-added boxes get full confidence
                    });

                    // Select the new box
                    selectedBox = correctedDetections.length - 1;
                    showFeedback('Box added', true);
                }

                isAddingBox = false;
            } else if (isEditingBox && selectedBox !== null) {
                // Update the box in corrected detections
                let x = currentBox.x;
                let y = currentBox.y;
                let width = currentBox.width;
                let height = currentBox.height;

                // Convert to API format
                let bbox = [
                    x, y, x + width, y + height
                ];

                // Scale to original image dimensions
                bbox = scaleBoxToOriginal(bbox);

                // Update box
                correctedDetections[selectedBox].bbox = bbox;
                showFeedback('Box updated', true);

                isEditingBox = false;
            }

            currentBox = null;
            updateBoxLists();
            drawBoxes();
        }

        function drawBoxes() {
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw all corrected boxes
            correctedDetections.forEach((box, index) => {
                const [x1, y1, x2, y2] = translateBox(box.bbox);

                let color = index === selectedBox ? '#e74c3c' : '#2ecc71';

                // Draw box
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                // Draw label
                ctx.fillStyle = color;
                ctx.font = '12px Arial';
                let label = `${box.class_name} (${(box.confidence * 100).toFixed(0)}%)`;
                ctx.fillRect(x1, y1 - 20, ctx.measureText(label).width + 10, 20);
                ctx.fillStyle = 'white';
                ctx.fillText(label, x1 + 5, y1 - 5);
            });

            // Draw current box if drawing
            if (isDrawing && currentBox) {
                ctx.strokeStyle = '#3498db';
                ctx.lineWidth = 2;

                if (isAddingBox) {
                    ctx.strokeRect(startX, startY, currentBox.width, currentBox.height);
                } else if (isEditingBox) {
                    ctx.strokeRect(currentBox.x, currentBox.y, currentBox.width, currentBox.height);
                }
            }
        }

        function translateBox(bbox) {
            // Scale bbox coordinates from original image to canvas
            const [x1, y1, x2, y2] = bbox;

            const scaleX = canvas.width / videoFeed.naturalWidth;
            const scaleY = canvas.height / videoFeed.naturalHeight;

            return [
                x1 * scaleX,
                y1 * scaleY,
                x2 * scaleX,
                y2 * scaleY
            ];
        }

        function scaleBoxToOriginal(bbox) {
            // Scale bbox coordinates from canvas to original image
            const [x1, y1, x2, y2] = bbox;

            const scaleX = videoFeed.naturalWidth / canvas.width;
            const scaleY = videoFeed.naturalHeight / canvas.height;

            return [
                Math.round(x1 * scaleX),
                Math.round(y1 * scaleY),
                Math.round(x2 * scaleX),
                Math.round(y2 * scaleY)
            ];
        }

        function getClassName(classId) {
            const classObj = classes.find(c => c.id === classId);
            return classObj ? classObj.name : `Class ${classId}`;
        }

        function showFeedback(message, isSuccess) {
            const feedback = document.getElementById('feedbackMessage');
            feedback.textContent = message;
            feedback.className = isSuccess ? 'feedback success' : 'feedback error';
            feedback.style.display = 'block';

            // Hide after 3 seconds
            setTimeout(() => {
                feedback.style.display = 'none';
            }, 3000);
        }

        // Set up video feed image loaded event
        videoFeed.onload = function() {
            resizeCanvas();
        };

        // Initialize on window load
        window.onload = init;
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    # This can be used to test the HTML template
    from flask import Flask

    app = Flask(__name__)


    @app.route('/')
    def index():
        return ANNOTATION_HTML


    app.run(debug=True)