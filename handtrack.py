import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import math
import threading
import os
import time

pyautogui.FAILSAFE = False

# Scroll tracking
previous_scroll_y = None
scroll_accumulator = 0
scroll_mode_active = False
previous_vulcan_state = False

# Threshold for distinguishing click and drag
CLICK_THRESHOLD = 0.3

# Threshold for detecting significant motion
MOTION_THRESHOLD = 7  # adjust based on sensitivity
SMOOTHING_ALPHA = 0.2  # weight for smoothing

# Mouse sensitivity
SCALING_FACTOR = 8.0

# Gesture cooldown times (seconds)
GESTURE_COOLDOWN = 0.4  # Time between gesture triggers

# Scroll gesture settings
SCROLL_ENABLED = True
SCROLL_THRESHOLD = 15  # Minimum pixel movement to trigger scroll
SCROLL_SPEED = 3  # Lines per scroll event

# Control toggles
mouse_control_enabled = True
scroll_control_enabled = True

# Camera orientation modes
ORIENTATION_FRONT = 'front'  # Palm facing camera (typical webcam)
ORIENTATION_TOP_DOWN = 'top_down'  # Camera mounted above, looking down at desk
current_orientation = ORIENTATION_FRONT  # Default mode

# Last gesture times for cooldown
last_gesture_times = {
    'index': 0,
    'middle': 0,
    'ring': 0,
    'pinky': 0,
    'scroll': 0
}

# Active gesture for visual feedback
active_gesture = None
gesture_display_time = 0

# Initialize the Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array ([[1, 0, 0, 0],
                                      [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Initialize MediaPipe gesture recognizer (new API)
base_options = mp.tasks.BaseOptions(model_asset_path='gesture_recognizer.task')
options = mp.tasks.vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)
gesture_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

# Set up pyautogui
screen_width, screen_height = pyautogui.size()

frame_counter = 0
frame_skip = 5
previous_cursor_position = (screen_width // 2, screen_height // 2)

cursor_position = None
hand_detected = False  # Track if hand is currently visible
last_hand_detected = False  # Previous frame's detection state

# Function to move the cursor in a separate thread
def move_cursor():
    while True:
        if hand_detected and cursor_position and mouse_control_enabled:
            pyautogui.moveTo(cursor_position[0], cursor_position[1])
        time.sleep(0.01)



cursor_thread = threading.Thread(target=move_cursor)
cursor_thread.daemon = True
cursor_thread.start()

# Helper function to calculate Euclidean distance
def calculate_distance(landmark1, landmark2, image_width, image_height):
    x1, y1 = int(landmark1.x * image_width), int(landmark1.y * image_height)
    x2, y2 = int(landmark2.x * image_width), int(landmark2.y * image_height)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Helper function to calculate distance ratios
def calculate_distance_ratios(dist_thumb_index, dist_thumb_middle, dist_thumb_ring, dist_thumb_pinky, palm_length):
    # Calculate the ratios of distances to thumb-index distance
    ratios = {
        "T->I / PL": dist_thumb_index / palm_length if palm_length != 0 else 1,  
        "T->M / PL": dist_thumb_middle / palm_length if palm_length != 0 else 1,
        "T->R / PL": dist_thumb_ring / palm_length if palm_length != 0 else 1,
        "T->P / PL": dist_thumb_pinky / palm_length if palm_length != 0 else 1, 
    }
    return ratios

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

def transform_coordinates_by_orientation(landmark, screen_width, screen_height, orientation):
    """Transform landmark coordinates based on camera orientation."""
    if orientation == ORIENTATION_FRONT:
        # Standard: palm facing camera
        screen_x = int(landmark.x * screen_width)
        screen_y = int(landmark.y * screen_height)
    else:  # ORIENTATION_TOP_DOWN
        # Top-down: camera above desk looking down
        # X stays the same, but Y interpretation changes
        screen_x = int(landmark.x * screen_width)
        # In top-down, forward/back motion is more natural for Y
        screen_y = int(landmark.y * screen_height)
    
    return screen_x, screen_y

def get_scroll_axis_by_orientation(landmark_5, h, w, orientation):
    """Get the appropriate coordinate for scrolling based on orientation."""
    if orientation == ORIENTATION_FRONT:
        # Front view: vertical hand movement = Y axis
        return landmark_5.y * h
    else:  # ORIENTATION_TOP_DOWN
        # Top-down view: forward/back hand movement = Y axis (depth)
        # Use Y coordinate which represents depth from camera perspective
        return landmark_5.y * h

def can_trigger_gesture(gesture_name):
    """Check if enough time has passed since last gesture trigger."""
    current_time = time.time()
    return (current_time - last_gesture_times.get(gesture_name, 0)) > GESTURE_COOLDOWN

def trigger_gesture(gesture_name, action_func, display_name):
    """Trigger a gesture with cooldown and visual feedback."""
    global active_gesture, gesture_display_time
    if can_trigger_gesture(gesture_name):
        action_func()
        last_gesture_times[gesture_name] = time.time()
        active_gesture = display_name
        gesture_display_time = time.time()
        print(f"{display_name}!")
        return True
    return False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip and convert to RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Process the frame with timestamp
    results = gesture_recognizer.recognize_for_video(mp_image, frame_counter)
    frame_counter += 1

    # Update hand detection status
    hand_detected = bool(results.hand_landmarks)
    
    # If hand just disappeared, center the cursor and release control
    if last_hand_detected and not hand_detected:
        pyautogui.moveTo(screen_width // 2, screen_height // 2)
        cursor_position = None
        print("Hand lost - cursor released to center")
    
    # If hand just appeared, resume control
    if not last_hand_detected and hand_detected:
        print("Hand detected - resuming control")
    
    last_hand_detected = hand_detected

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Get coordinates for landmarks
            landmark_0 = hand_landmarks[0] # wrist
            landmark_5 = hand_landmarks[5] # First knuckle, first finger
            landmark_4 = hand_landmarks[4] # thumb tip
            landmark_8 = hand_landmarks[8] # index tip
            landmark_12 = hand_landmarks[12] # middle tip
            landmark_16 = hand_landmarks[16] # ring tip
            landmark_20 = hand_landmarks[20] # pinky tip
            h, w, _ = image.shape
            dist_thumb_index = round(calculate_distance(landmark_4, landmark_8, w, h), 1)
            dist_thumb_middle = round(calculate_distance(landmark_4, landmark_12, w, h), 1)
            dist_thumb_ring = round(calculate_distance(landmark_4, landmark_16, w, h), 1)
            dist_thumb_pinky = round(calculate_distance(landmark_4, landmark_20, w, h), 1)
            palm_length = round(calculate_distance(landmark_5, landmark_0, w, h),1)
        
            # Calculate the finger distance ratios
            ratios = calculate_distance_ratios(dist_thumb_index, dist_thumb_middle, dist_thumb_ring, dist_thumb_pinky, palm_length)

            # Get recognized gestures
            detected_gesture = None
            if results.gestures:
                detected_gesture = results.gestures[0][0].category_name
            
            # Scroll gesture detection: Thumb_Up = scroll up, Victory = scroll down
            if detected_gesture and scroll_control_enabled and SCROLL_ENABLED:
                if detected_gesture == "Thumb_Up" and can_trigger_gesture('scroll_up'):
                    pyautogui.scroll(SCROLL_SPEED * 2)
                    last_gesture_times['scroll_up'] = time.time()
                    active_gesture = "Scroll Up 👍"
                    gesture_display_time = time.time()
                    print("Scrolling Up")
                elif detected_gesture == "Victory" and can_trigger_gesture('scroll_down'):
                    pyautogui.scroll(-SCROLL_SPEED * 2)
                    last_gesture_times['scroll_down'] = time.time()
                    active_gesture = "Scroll Down ✌️"
                    gesture_display_time = time.time()
                    print("Scrolling Down")

            # Pinch gesture detection (only if not doing scroll gestures)
            if detected_gesture not in ["Thumb_Up", "Victory"] and mouse_control_enabled:
                if ratios["T->I / PL"] < 0.15:  # Thumb and index
                    trigger_gesture('index', pyautogui.mouseDown, "Mouse Down (Drag)")
                elif ratios["T->M / PL"] < 0.15:  # Thumb and middle
                    trigger_gesture('middle', pyautogui.mouseUp, "Mouse Up")
                elif ratios["T->R / PL"] < 0.15:  # Thumb and ring
                    trigger_gesture('ring', pyautogui.click, "Left Click")
                elif ratios["T->P / PL"] < 0.15:  # Thumb and pinky
                    trigger_gesture('pinky', pyautogui.rightClick, "Right Click")

            # Get cursor position with orientation-aware transformation
            screen_x, screen_y = transform_coordinates_by_orientation(landmark_5, screen_width, screen_height, current_orientation)
            
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
            if mouse_control_enabled:
                cursor_position = stabilized_position
            previous_cursor_position = stabilized_position

            h, w, _ = image.shape
            cx, cy = int(landmark_5.x * w), int(landmark_5.y * h)
            
            # Draw hand landmarks with color based on gesture state
            current_time = time.time()
            gesture_active_now = active_gesture and (current_time - gesture_display_time) < 0.5
            landmark_color = (0, 255, 255) if gesture_active_now else (0, 255, 0)  # Yellow if gesture active
            
            for idx, landmark in enumerate(hand_landmarks):
                cx_lm, cy_lm = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (cx_lm, cy_lm), 5, landmark_color, -1)
            
            # Status display
            status_y = 30
            
            # Orientation mode
            orientation_text = "Front" if current_orientation == ORIENTATION_FRONT else "Top-Down"
            cv2.putText(image, f"Mode: {orientation_text}", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            status_y += 30
            mouse_status = "ON" if mouse_control_enabled else "OFF"
            mouse_color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
            cv2.putText(image, f"Mouse: {mouse_status}", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouse_color, 2)
            
            status_y += 30
            scroll_status = "ON" if scroll_control_enabled else "OFF"
            scroll_color = (0, 255, 0) if scroll_control_enabled else (0, 0, 255)
            cv2.putText(image, f"Scroll: {scroll_status}", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, scroll_color, 2)
            
            status_y += 30
            cv2.putText(image, f"Sensitivity: {SCALING_FACTOR:.1f}x", (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Gesture feedback (large, centered)
            if gesture_active_now:
                text_size = cv2.getTextSize(active_gesture, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h - 50
                # Background rectangle
                cv2.rectangle(image, (text_x - 10, text_y - text_size[1] - 10), 
                            (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                cv2.putText(image, active_gesture, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            
            # Show detected gesture
            if detected_gesture:
                gesture_text = f"Gesture: {detected_gesture}"
                gesture_color = (0, 255, 255)
            else:
                gesture_text = "No Gesture"
                gesture_color = (255, 255, 255)
            
            cv2.putText(image, gesture_text, (w - 350, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
                
    # Show help text if no hand detected
    if not results.hand_landmarks:
        # Center message
        center_msg = "NO HAND DETECTED - CURSOR RELEASED"
        text_size = cv2.getTextSize(center_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = 100
        cv2.rectangle(image, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
        cv2.putText(image, center_msg, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        help_text = [
            "Controls:",
            "O - Toggle Orientation (Front/Top-Down)",
            "T - Toggle Mouse Control",
            "S - Toggle Scroll",
            "+ - Increase Sensitivity",
            "- - Decrease Sensitivity",
            "ESC - Quit",
            "",
            f"Current Mode: {'Front' if current_orientation == ORIENTATION_FRONT else 'Top-Down'}"
        ]
        for i, text in enumerate(help_text):
            cv2.putText(image, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('MediaPipe Hands', image)
    key = cv2.waitKey(5) & 0xFF
    
    # Keyboard controls
    if key == 27:  # ESC
        break
    elif key == ord('o') or key == ord('O'):
        current_orientation = ORIENTATION_TOP_DOWN if current_orientation == ORIENTATION_FRONT else ORIENTATION_FRONT
        mode_name = "Front-Facing" if current_orientation == ORIENTATION_FRONT else "Top-Down"
        print(f"Orientation Mode: {mode_name}")
    elif key == ord('t') or key == ord('T'):
        mouse_control_enabled = not mouse_control_enabled
        print(f"Mouse Control: {'ENABLED' if mouse_control_enabled else 'DISABLED'}")
    elif key == ord('s') or key == ord('S'):
        scroll_control_enabled = not scroll_control_enabled
        print(f"Scroll Control: {'ENABLED' if scroll_control_enabled else 'DISABLED'}")
    elif key == ord('+') or key == ord('='):
        SCALING_FACTOR = min(SCALING_FACTOR + 1.0, 20.0)
        print(f"Sensitivity increased to {SCALING_FACTOR:.1f}x")
    elif key == ord('-') or key == ord('_'):
        SCALING_FACTOR = max(SCALING_FACTOR - 1.0, 2.0)
        print(f"Sensitivity decreased to {SCALING_FACTOR:.1f}x")

cap.release() 
