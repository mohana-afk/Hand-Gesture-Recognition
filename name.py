import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import numpy as np

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# Global variable for latest result
latest_result = None

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def draw_landmarks(image, landmarks, connections):
    h, w, _ = image.shape
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), 5, (255, 255, 0), -1)
    for conn in connections:
        start = landmarks[conn[0]]
        end = landmarks[conn[1]]
        start_x, start_y = int(start.x * w), int(start.y * h)
        end_x, end_y = int(end.x * w), int(end.y * h)
        cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 100, 0), 2)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 680)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def save_landmarks(landmarks, label):
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    row.append(label)

    with open("landmarks.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect async
        landmarker.detect_async(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Draw if result available
        if latest_result:
            for hand_landmarks in latest_result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)
                # Save landmarks, perhaps on key press or always
                # For now, save every frame if hand detected
                save_landmarks(hand_landmarks, "unknown")  # Change label as needed

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if latest_result and latest_result.hand_landmarks:
            hand_points = latest_result.hand_landmarks[0]
            if key == ord('0'):
                save_landmarks(hand_points, "fist")
            elif key == ord('1'):
                save_landmarks(hand_points, "middle")
            elif key == ord('2'):
                save_landmarks(hand_points, "yoo")
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()