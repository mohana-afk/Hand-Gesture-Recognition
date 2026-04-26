import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Load model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

gesture_names = {0: "Fist", 1: "Middle", 2: "Yoo"}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # The prediction is already the string "fist", "middle", etc. Make it capitalized.
            prediction = model.predict([landmarks])[0]
            gesture = str(prediction).capitalize()

            cv2.putText(frame, gesture, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 100, 0), 3)

            # Check and print the gesture
            if gesture == "Fist":
                print("Fist")

            elif gesture == "Middle" :
                print("Middle")
                
            elif gesture == "Yoo":
                print("Yoo")
                break

    cv2.imshow("FRAME", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()