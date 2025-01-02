import cv2
import numpy as np
import pyautogui
import keyboard
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
camera_index = int(os.getenv('CAMERA_INDEX'))
lower_color = np.array([int(os.getenv('LOWER_COLOR_H')), int(os.getenv('LOWER_COLOR_S')), int(os.getenv('LOWER_COLOR_V'))])
upper_color = np.array([int(os.getenv('UPPER_COLOR_H')), int(os.getenv('UPPER_COLOR_S')), int(os.getenv('UPPER_COLOR_V'))])

# Open the camera
cap = cv2.VideoCapture(camera_index)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Center the cursor on start
pyautogui.moveTo(screen_width // 2, screen_height // 2)

# disable PyAutoGUI failsafe
pyautogui.FAILSAFE = False

# Variable to track the state of tracking
tracking_enabled = False
handtrack_quit = False

def toggle_tracking():
    global tracking_enabled
    tracking_enabled = not tracking_enabled
    print(f"Tracking {'enabled' if tracking_enabled else 'disabled'}")

# Set up the shortcut key (e.g., 't' key) to toggle tracking
keyboard.add_hotkey('windows+alt+x', toggle_tracking)

# Sensitivity factor
#sensitivity = 2.0

# Smoothing variables
prev_x, prev_y = 0, 0
alpha = 0.5 # smoothing factor


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if tracking_enabled:
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, draw a circle around the largest one and move the cursor
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(frame, center, radius, (0, 255, 0), 2)

            # Map the center coordinates to screen coordinates
            screen_x = int(screen_width * (x / frame.shape[1]))
            screen_y = int(screen_height * (y / frame.shape[0]))

            # Apply smoothing
            screen_x = int(alpha * screen_x + (1 - alpha) * prev_x)
            screen_y = int(alpha * screen_y + (1 - alpha) * prev_y)
            prev_x, prev_y = screen_x, screen_y

            # Move the cursor
            pyautogui.moveTo(screen_x, screen_y)
        else:
            #Move the cursor to the center of the screen if the dot is not detected
            pyautogui.moveTo(screen_width // 2, screen_height // 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
