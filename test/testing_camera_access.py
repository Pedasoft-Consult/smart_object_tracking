import cv2


def test_camera(index):
    video_path = "videos/person-bicycle-car-detection.mp4"
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop back to beginning if video ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Process frame here

        # Display frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Test camera indices from 0 to 3
for i in range(4):
    test_camera(i)