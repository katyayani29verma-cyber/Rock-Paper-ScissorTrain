import cv2
import mediapipe as mp
# import csv

mp_hands = mp.solutions.hands(
 max_num_hands = 1,
 min_detection_confidence = 0.7,
 min_tracking_confidence = 0.7
)
mp_draw = mp.solutions.utils
cap = cv2.VideoCapture(0)
gesture = input("Enter gesture label (ROCK / PAPER / SCISSOR): ")
file = open("gesture_data.csv", "a", newline="")
writer = csv.writer(file)
