# tracker_visualizer.py - visualization and tracking output
import cv2
import csv
import os
import logging

COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 128, 0), (0, 128, 128),
    (200, 200, 50), (50, 200, 200), (200, 50, 200), (100, 255, 100)
]

CSV_HEADER = ['frame', 'track_id', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2']

_csv_initialized = False
_csv_file = None
_csv_writer = None


def draw_and_log_tracks(display_frame, tracks, model, frame_count, output_dir="output"):
    global _csv_initialized, _csv_file, _csv_writer

    logger = logging.getLogger('TrackerVisualizer')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not _csv_initialized:
        csv_path = os.path.join(output_dir, "tracking_results.csv")
        write_header = not os.path.exists(csv_path)
        _csv_file = open(csv_path, 'a', newline='')
        _csv_writer = csv.writer(_csv_file)
        if write_header:
            _csv_writer.writerow(CSV_HEADER)
        _csv_initialized = True

    for track in tracks:
        try:
            # Handle different track object formats
            if hasattr(track, 'bbox'):
                # Direct bbox attribute
                if callable(track.bbox):
                    bbox = track.bbox()
                else:
                    bbox = track.bbox
                x1, y1, x2, y2 = map(int, bbox)
                track_id = getattr(track, 'track_id', 0)
                class_id = getattr(track, 'class_id', 0)
                confidence = getattr(track, 'confidence', 0.0)

            elif hasattr(track, 'tlbr'):
                # Handle TLBR format
                x1, y1, x2, y2 = map(int, track.tlbr)
                track_id = getattr(track, 'track_id', getattr(track, 'id', 0))
                class_id = getattr(track, 'class_id', 0)
                confidence = getattr(track, 'confidence', 0.0)

            elif hasattr(track, 'tlwh'):
                # Handle TLWH format
                x, y, w, h = map(int, track.tlwh)
                x1, y1, x2, y2 = x, y, x + w, y + h
                track_id = getattr(track, 'track_id', getattr(track, 'id', 0))
                class_id = getattr(track, 'class_id', 0)
                confidence = getattr(track, 'confidence', 0.0)

            elif isinstance(track, dict):
                # Dictionary format
                bbox = track.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, bbox)
                track_id = track.get('id', track.get('track_id', 0))
                class_id = track.get('class_id', 0)
                confidence = track.get('confidence', 0.0)

            else:
                # Skip this track if we can't determine the format
                logger.warning(f"Unknown track format: {type(track)}")
                continue

            # Get class name
            class_name = model.names[class_id] if hasattr(model,
                                                          'names') and class_id in model.names else f"class_{class_id}"
            color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

            # Draw on frame
            label = f"{class_name} ID:{track_id}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trail if available
            if hasattr(track, 'trail') and len(track.trail) > 1:
                for i in range(1, len(track.trail)):
                    pt1 = (int(track.trail[i - 1][0]), int(track.trail[i - 1][1]))
                    pt2 = (int(track.trail[i][0]), int(track.trail[i][1]))
                    cv2.line(display_frame, pt1, pt2, color, 1)

            # Write to CSV
            _csv_writer.writerow([frame_count, track_id, class_id, class_name, confidence, x1, y1, x2, y2])

        except Exception as e:
            logger.error(f"Error processing track: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return display_frame


def finalize_visualizer():
    global _csv_file
    if _csv_file:
        _csv_file.close()
        print("Tracking data exported and CSV closed.")
        return True
    return False