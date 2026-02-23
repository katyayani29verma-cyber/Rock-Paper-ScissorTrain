import cv2
import mediapipe as mp
import pandas as pd
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =============================
# Create / Load Dataset
# =============================
file_name = "dataset.csv"

if not os.path.exists(file_name):
    columns = []
    for i in range(21):
        columns += [f"x{i}", f"y{i}", f"z{i}"]
    columns.append("label")
    df = pd.DataFrame(columns=columns)
else:
    df = pd.read_csv(file_name)

# =============================
# Ask for Label
# =============================
label = input("Enter label (rock/paper/scissors): ").lower()

if label not in ["rock", "paper", "scissors"]:
    print("Invalid label.")
    exit()

# =============================
# Start Webcam
# =============================
cap = cv2.VideoCapture(0)

print("Press SPACE to capture sample")
print("Press Q to quit")

sample_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            data = []
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                # Normalize relative to wrist
                data.append(lm.x - wrist.x)
                data.append(lm.y - wrist.y)
                data.append(lm.z - wrist.z)

    cv2.putText(frame, f"Samples: {sample_count}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1)

    # Capture when SPACE pressed
    if key == 32 and results.multi_hand_landmarks:
        data.append(label)
        df.loc[len(df)] = data
        sample_count += 1
        print(f"Captured {sample_count}")

        time.sleep(0.3)  # small delay to avoid duplicates

    # Quit when Q pressed
    if key & 0xFF == ord('q'):
        break

# =============================
# Save Dataset
# =============================
df.to_csv(file_name, index=False)

cap.release()
cv2.destroyAllWindows()

print("Dataset saved successfully.")