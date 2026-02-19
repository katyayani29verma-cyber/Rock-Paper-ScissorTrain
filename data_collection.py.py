import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter label (rock/paper/scissors): ")

if not os.path.exists("dataset.csv"):
    df = pd.DataFrame()
else:
    df = pd.read_csv("dataset.csv")

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data.append(label)

            df.loc[len(df)] = data

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df.to_csv("dataset.csv", index=False)
cap.release()
cv2.destroyAllWindows()
