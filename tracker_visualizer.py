# Fixed detect_and_track.py - finalized visualization and tracking output
import cv2
import csv
import os

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
            x1, y1, x2, y2 = map(int, track.bbox)
            track_id = track.track_id
            class_id = track.class_id
            confidence = track.confidence
            class_name = model.names[class_id] if hasattr(model, 'names') and class_id in model.names else f"class_{class_id}"
            color = COLOR_PALETTE[class_id % len(COLOR_PALETTE)]

            label = f"{class_name} ID:{track_id}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if hasattr(track, 'trail') and len(track.trail) > 1:
                for i in range(1, len(track.trail)):
                    pt1 = (int(track.trail[i - 1][0]), int(track.trail[i - 1][1]))
                    pt2 = (int(track.trail[i][0]), int(track.trail[i][1]))
                    cv2.line(display_frame, pt1, pt2, color, 1)

            _csv_writer.writerow([frame_count, track_id, class_id, class_name, confidence, x1, y1, x2, y2])

        except Exception as e:
            print(f"Error processing track: {e}")


def finalize_visualizer():
    global _csv_file
    if _csv_file:
        _csv_file.close()
        print("Tracking data exported and CSV closed.")