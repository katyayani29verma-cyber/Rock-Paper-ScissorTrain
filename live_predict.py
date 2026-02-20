import cv2
import mediapipe as mp
import joblib
import numpy as np

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

# =============================
# Start Webcam
# =============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction_text = "No Hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            data = []
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                # Same normalization used during training
                data.append(lm.x - wrist.x)
                data.append(lm.y - wrist.y)
                data.append(lm.z - wrist.z)

            data = np.array(data).reshape(1, -1)

            prediction = model.predict(data)
            prediction_text = prediction[0]

    # Display Prediction
    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Live Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()