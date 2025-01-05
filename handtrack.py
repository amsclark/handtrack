import cv2
import numpy as np
import pyautogui
import keyboard
import mediapipe as mp
import math
import threading
import os
import time

pyautogui.FAILSAFE = False

global gesture_active 
global last_gesture_active
gesture_active = False
last_gesture_active = False
gesture_start_time = None
dragging = False

# Threshold for distinguishing click and drag
CLICK_THRESHOLD = 0.3

#threshold for detecting significant motion
MOTION_THRESHOLD = 7 #adjust based on sensitivity
SMOOTHING_ALPHA = 0.2 # weight for smoothing

# Mouse sensitivity
SCALING_FACTOR = 2.0

# Initialize the Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array ([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up pyautogui
screen_width, screen_height = pyautogui.size()

frame_counter = 0
frame_skip = 5
previous_cursor_position = (screen_width // 2, screen_height // 2)

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

# Function to calculate the scaled position
def calculate_scaled_position(current_pos, prev_pos, scaling_factor):
    if prev_pos is None:  # Initialize on the first frame
        return current_pos
    delta_x = (current_pos[0] - prev_pos[0]) * scaling_factor
    delta_y = (current_pos[1] - prev_pos[1]) * scaling_factor
    return int(prev_pos[0] + delta_x), int(prev_pos[1] + delta_y)


# Function to apply scaling directly to raw smoothed coordinates
def apply_scaling(smoothed_x, smoothed_y, scaling_factor):
    center_x, center_y = screen_width // 2, screen_height // 2
    delta_x = (smoothed_x - center_x) * scaling_factor
    delta_y = (smoothed_y - center_y) * scaling_factor
    return int(center_x + delta_x), int(center_y + delta_y)

# Stabilization helper functions
def apply_threshold(current_pos, prev_pos, threshold=MOTION_THRESHOLD):
    """Ignores small movements below a certain threshold."""
    if prev_pos is None:  # If it's the first frame
        return current_pos
    dist = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
    if dist < threshold:
        return prev_pos
    return current_pos

def smooth_position(current_pos, prev_pos, alpha=SMOOTHING_ALPHA):
    """Applies weighted averaging to smooth motion."""
    if prev_pos is None:  # If it's the first frame
        return current_pos
    return tuple((1 - alpha) * np.array(prev_pos) + alpha * np.array(current_pos))


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

                    # Get coordinates for landmarks
                    landmark_5 = hand_landmarks.landmark[5] # First knuckle, first finger
                    landmark_4 = hand_landmarks.landmark[4] # thumb tip
                    landmark_8 = hand_landmarks.landmark[8] # index tip
                    landmark_12 = hand_landmarks.landmark[12] # middle tip
                    landmark_16 = hand_landmarks.landmark[16] # ring tip
                    landmark_20 = hand_landmarks.landmark[20] # pinky tip
                    h, w, _ = image.shape
                    dist_thumb_index = round(calculate_distance(landmark_4, landmark_8, w, h), 1)
                    dist_thumb_middle = round(calculate_distance(landmark_4, landmark_12, w, h), 1)
                    dist_thumb_ring = round(calculate_distance(landmark_4, landmark_16, w, h), 1)
                    dist_thumb_pinky = round(calculate_distance(landmark_4, landmark_20, w, h), 1)
                    

                    normalized_x, normalized_y = landmark_5.x, landmark_5.y
                    screen_x = int(normalized_x * screen_width)
                    screen_y = int(normalized_y * screen_height)
                    
                    # Kalman filter: Update with new measurement
                    measurement = np.array([[np.float32(screen_x)], [np.float32(screen_y)]])
                    kalman.correct(measurement)

                    # Kalman filter: Predict the smoothed position
                    predicted = kalman.predict()
                    smoothed_x, smoothed_y = int(predicted[0]), int(predicted[1])

                    scaled_position = apply_scaling(smoothed_x, smoothed_y, SCALING_FACTOR) 


                    # Stabilize cursor movement
                    stabilized_position = apply_threshold(scaled_position, previous_cursor_position)
                    stabilized_position = smooth_position(stabilized_position, previous_cursor_position)


                    # Move the cursor to the smoothed position
                    cursor_position = stabilized_position
                    previous_cursor_position = stabilized_position

                    h, w, _ = image.shape
                    cx, cy = int(landmark_5.x * w), int(landmark_5.y * h)
                    
                    cv2.putText(image, f"Landmark 5: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(image, f"T<->I Dist: {dist_thumb_index}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(image, f"T<->M Dist: {dist_thumb_middle}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(image, f"T<->R Dist: {dist_thumb_ring}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(image, f"T<->P Dist: {dist_thumb_pinky}", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release() 
