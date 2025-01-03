import cv2
import numpy as np
import pyautogui
#from pynput.mouse import Button, Controller
import keyboard
import mediapipe as mp
import math
import threading
import os
import time

pyautogui.FAILSAFE = False

gesture_active = False
gesture_start_time = None
dragging = False

# Threshold for distinguishing click and drag
CLICK_THRESHOLD = 0.3



# Initialize the Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array ([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#mouse = Controller()
screen_width, screen_height = pyautogui.size()

frame_counter = 0
frame_skip = 5
previous_cursor_position = None

cursor_position = None

# Function to move the cursor in a seperate thread
def move_cursor():
    while True:
        if cursor_position: 
            pyautogui.moveTo(cursor_position[0], cursor_position[1])

cursor_thread = threading.Thread(target=move_cursor)
cursor_thread.daemon = True
cursor_thread.start()

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
                    
                    # Kalman filter: Update with new measurement
                    measurement = np.array([[np.float32(screen_x)], [np.float32(screen_y)]])
                    kalman.correct(measurement)

                    # Kalman filter: Predict the smoothed position
                    predicted = kalman.predict()
                    smoothed_x, smoothed_y = int(predicted[0]), int(predicted[1])

                    # Move the cursor to the smoothed position
                    # cursor_position = (screen_x, screen_y)
                    #pyautogui.moveTo(screen_x, screen_y)
                    cursor_position = (smoothed_x, smoothed_y)

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
                        if not gesture_active:
                            gesture_active = True
                            gesture_start_time = time.time()
                        if time.time() - gesture_start_time > CLICK_THRESHOLD and not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                    else:
                        if gesture_active:
                            gesture_active = False
                            gesture_duration = time.time() - gesture_start_time

                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False
                            elif gesture_duration <= CLICK_THRESHOLD:
                                pyautogui.click()
                                gesture_text = "Click"
                                cv2.putText(image, gesture_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release() 
