import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
from pythonlogic import decide_winner

# =============================
# Load Trained Model
# =============================
model = joblib.load("rps_model.pkl")

# =============================
# Initialize MediaPipe
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# =============================
# Game Timer Settings
# =============================
round_duration = 3      # seconds to wait before locking move
result_display_time = 5  # seconds to show result

round_start_time = time.time()
round_locked = False

player_move = ""
computer_move = ""
result = ""

# =============================
# Main Loop
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()
    elapsed_time = current_time - round_start_time
    time_left = max(0, round_duration - int(elapsed_time))

    # =============================
    # Countdown Phase
    # =============================
    if not round_locked:

        cv2.putText(frame, f"Show your move!",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        cv2.putText(frame, f"Capturing in: {time_left}s",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

        if time_left == 0 and results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                data = []
                wrist = hand_landmarks.landmark[0]

                for lm in hand_landmarks.landmark:
                    data.append(lm.x - wrist.x)
                    data.append(lm.y - wrist.y)
                    data.append(lm.z - wrist.z)

                data = np.array(data).reshape(1, -1)

                prediction = model.predict(data)
                player_move = prediction[0]

                computer_move, result = decide_winner(player_move)

                round_locked = True
                round_start_time = time.time()

    # =============================
    # Result Display Phase
    # =============================
    else:

        cv2.putText(frame, f"Player: {player_move}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame, f"Computer: {computer_move}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.putText(frame, f"Result: {result}",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # Reset after showing result
        if elapsed_time > result_display_time:
            round_locked = False
            round_start_time = time.time()
            player_move = ""
            computer_move = ""
            result = ""

    cv2.imshow("Live Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()