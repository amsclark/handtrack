import cv2
import numpy as np
import pyautogui
#from pynput.mouse import Button, Controller
import keyboard
import mediapipe as mp
import math
import os


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Helper function to calculate Euclidean distance
def calculate_distance(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
    x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")

            
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # Get coordinates for first knuckle, first finger
                    landmark_5 = hand_landmarks.landmark[5]
                    normalized_x, normalized_y = landmark_5.x, landmark_5.y
                    screen_x = int(normalized_x * screen_width)
                    screen_y = int(normalized_y * screen_height)
                    pyautogui.moveTo(screen_x, screen_y)

                    h, w, _ = image.shape
                    cx, cy = int(landmark_5.x * w), int(landmark_5.y * h)
                    
                    cv2.putText(image, f"Landmark 5: ({cx}, {cy}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get landmarks for the thumb tip and first fingertip
                    thumb_tip = hand_landmarks.landmark[4]
                    index_tip = hand_landmarks.landmark[8]
                    distance = calculate_distance(thumb_tip, index_tip, w, h)
                    
                    threshold = 20

                    if distance < threshold:
                        #mouse.click(Button.left, 1)
                        pyautogui.click()
                        gesture_text = "Click"
                        cv2.putText(image, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
cap.release() 
