# HAND TRACKING MOUSE CONTROL

This script uses your webcam and MediaPipe's Hand Landmarker to track your hand gestures and control the mouse cursor. Move the cursor by moving your hand, and perform clicks using pinch gestures with different finger combinations.

## FEATURES

### Cursor Control
- **Hand Position Tracking** - Move your hand to control the cursor position
- **Automatic Release** - Cursor releases to center when hand not visible (allows regular mouse use)
- **Dual Orientation Modes** - Front-facing (palm) or top-down (desk) camera setups
- **Kalman Filtering** - Smooth tracking with reduced jitter
- **Motion Threshold** - Ignores micro-movements for stability
- **Scaling Factor** - Amplifies hand movements for easier cursor control across large screens

### Gesture Controls

**Pinch Gestures** (thumb to specific fingers):

| Gesture | Action |
|---------|--------|
| **Thumb + Index** (pinch) | Mouse Down (start drag) |
| **Thumb + Middle** (pinch) | Mouse Up (end drag) |
| **Thumb + Ring** (pinch) | Left Click |
| **Thumb + Pinky** (pinch) | Right Click |

**Hand Shape Gestures:**

| Gesture | Action |
|---------|--------|
| **Vulcan Salute 🖖** (index+middle together, ring+pinky together, gap between) + move | Scroll up/down |

The Vulcan salute is a distinctive hand gesture where you extend all four fingers with a V-shaped split between the middle and ring fingers. This unambiguous gesture prevents accidental scroll triggering.

### Keyboard Controls

| Key | Action |
|-----|--------|
| **O** | Toggle Orientation (Front/Top-Down) |
| **T** | Toggle Mouse Control ON/OFF |
| **S** | Toggle Scroll Control ON/OFF |
| **+** | Increase Sensitivity |
| **-** | Decrease Sensitivity |
| **ESC** | Quit |

### Camera Orientation Modes

**Front Mode (Default)** - Typical webcam setup
- Palm facing camera
- Hand held 1-2 feet away
- Standard for most users

**Top-Down Mode** - Desk setup
- Camera mounted above, looking down
- Hands resting on desk
- Sees back of hands
- Optimized coordinate mapping

Switch modes anytime with **'O'** key

### Smart Features

- **Gesture Cooldowns** - 0.4s between gestures prevents accidental double-clicks
- **Visual Feedback** - Hand landmarks change color when gestures trigger
- **On-Screen Status** - Shows current mode, mouse/scroll status, sensitivity
- **Hand Detection** - When hand leaves view, cursor releases to center (physical mouse works)
- **Large Gesture Display** - Gesture name appears prominently when triggered

### Stabilization Features
- **Kalman Filter** - Predicts smooth cursor movement
- **Motion Threshold** - Ignores movements below threshold (reduces jitter)
- **Exponential Smoothing** - Weighted averaging for fluid motion
- **Gesture Cooldown** - Prevents rapid re-triggering of gestures
- **Scaling** - Amplifies small hand movements for better control

## REQUIREMENTS

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

## INSTALLATION

### Linux (Ubuntu/Debian)

1. **Navigate to your virtual environment (if not already created):**
```bash
cd eyetracker
source bin/activate
```

2. **Install required packages (if not already installed):**
```bash
pip install opencv-python mediapipe numpy pyautogui
```

3. **Download the MediaPipe hand model:**
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### Windows/Mac

Follow similar steps using a virtual environment and install the same packages. Download the hand_landmarker.task model to the same directory as the script.

## USAGE

1. **Activate the virtual environment:**
```bash
source bin/activate
```

2. **Run the hand tracking script:**
```bash
python3 handtrack.py
```

3. **Choose your camera setup:**
   - **Front-facing (default):** Hold hand up with palm facing camera
   - **Top-down:** Mount camera above desk, rest hands on surface
   - Press **'O'** to toggle between modes

4. **Control the cursor:**
   - Move your hand to move the cursor
   - Pinch gestures trigger clicks (see gesture table above)
   - **Vulcan salute (🖖)** and move up/down to scroll
     - Extend 4 fingers: index+middle together, ring+pinky together, gap between
   - When hand leaves view, cursor releases to center

5. **Adjust as needed:**
   - **'T'** - Toggle mouse control on/off
   - **'S'** - Toggle scroll on/off
   - **'+'** / **'-'** - Adjust sensitivity
   - **'O'** - Switch camera orientation mode

6. **Exit:**
   - Press **ESC** key to quit

## CONFIGURATION

Edit these parameters at the top of `handtrack.py`:

```python
# Motion sensitivity
MOTION_THRESHOLD = 7        # Ignore movements below this (pixels)
SMOOTHING_ALPHA = 0.2       # Smoothing weight (0.1-0.5, lower = smoother)

# Cursor scaling
SCALING_FACTOR = 8.0        # Amplifies hand movements (higher = more sensitive)

# Gesture cooldown
GESTURE_COOLDOWN = 0.4      # Seconds between gesture triggers

# Scroll settings
SCROLL_ENABLED = True       # Enable scroll gesture
SCROLL_THRESHOLD = 15       # Minimum movement to trigger scroll (pixels)
SCROLL_SPEED = 3            # Lines per scroll event
```

### Camera Orientation

Toggle between modes with **'O'** key or set default:

```python
current_orientation = ORIENTATION_FRONT      # Default: palm facing camera
# current_orientation = ORIENTATION_TOP_DOWN # Alternative: camera above desk
```

### Gesture Sensitivity

To adjust pinch detection sensitivity, modify the ratio thresholds in the gesture detection section:

```python
if ratios["T->I / PL"] < 0.15:  # Change 0.15 to adjust sensitivity
```

- **Lower values** (e.g., 0.10) = fingers must be closer together (harder to trigger)
- **Higher values** (e.g., 0.20) = more sensitive (easier to trigger accidentally)

## HOW IT WORKS

1. **Hand Detection** - MediaPipe detects 21 hand landmarks in real-time
2. **Hand Tracking** - Monitors hand presence; releases cursor when hand not visible
3. **Position Tracking** - Tracks the first knuckle of the index finger (landmark 5)
4. **Orientation Adjustment** - Applies coordinate transformation based on camera mode
5. **Kalman Filtering** - Smooths the position data to reduce jitter
6. **Scaling** - Amplifies hand movements relative to screen center
7. **Gesture Recognition** - Calculates distances between thumb and other fingertips
8. **Vulcan Salute Detection** - Recognizes V-shaped finger split for scroll mode
9. **Distance Ratios** - Normalizes distances by palm length for consistent detection
10. **Cooldown System** - Prevents rapid re-triggering of same gesture
11. **Click Actions** - Triggers mouse actions when distance ratios fall below threshold

## DISPLAY

The window shows:

**Status Overlay:**
- **Mode** - Current camera orientation (Front/Top-Down)
- **Mouse** - Control status (ON/OFF) in green/red
- **Scroll** - Scroll control status (ON/OFF) in green/red
- **Sensitivity** - Current scaling factor

**Visual Feedback:**
- **Green landmarks** - Normal tracking
- **Yellow landmarks** - Gesture actively triggered
- **Large centered text** - Current gesture name
- **"🖖 VULCAN (Scroll Mode)"** indicator in cyan when Vulcan salute detected
- **"CLOSED (Click Mode)"** indicator in white for normal operation
- **Red banner** - "NO HAND DETECTED" when hand not visible

**Help Text:**
- Displays all keyboard controls when no hand detected

## TROUBLESHOOTING

**Hand not detected:**
- Ensure good lighting conditions
- Move hand closer to the camera
- Make sure entire hand is visible in frame
- Check that camera is working: `ls /dev/video*`

**Cursor too jumpy:**
- Decrease `SMOOTHING_ALPHA` (try 0.1-0.15)
- Increase `MOTION_THRESHOLD` (try 10-15)
- Increase Kalman filter process noise: modify `kalman.processNoiseCov`

**Cursor moves too slowly:**
- Increase `SCALING_FACTOR` (try 10-15)
- Move hand closer to camera

**Cursor moves too fast:**
- Decrease `SCALING_FACTOR` (try 5-6)
- Move hand farther from camera

**Gestures trigger accidentally:**
- Decrease ratio thresholds (try 0.12 instead of 0.15)
- Keep fingers more spread apart when not clicking
- Cooldown system should prevent most accidents

**Gestures don't trigger:**
- Increase ratio thresholds (try 0.18-0.20)
- Pinch fingers closer together
- Ensure palm is visible for accurate palm length calculation
- Check that gesture cooldown has passed (0.4s default)

**Scroll not working:**
- Make sure you're doing the Vulcan salute correctly (🖖)
  - All 4 fingers extended
  - Index+middle together, ring+pinky together
  - Clear gap between middle and ring fingers
- Press 'S' to ensure scroll control is enabled
- Move hand more deliberately up/down
- Adjust `SCROLL_THRESHOLD` to be more sensitive

**Cursor "stuck" when hand removed:**
- This is now fixed - cursor should release to center automatically
- Check that `hand_detected` logic is working
- Restart the script if issue persists

**Wrong orientation for camera setup:**
- Press 'O' to toggle between Front and Top-Down modes
- Front mode: palm facing camera
- Top-Down mode: camera above desk looking down

**Mouse control interferes when switching to physical mouse:**
- Press 'T' to disable mouse control temporarily
- Remove hand from view to auto-release cursor
- Cursor automatically moves to center when hand not detected

**Wrong camera:**
- Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` or another index

**Script crashes on start:**
- Verify `hand_landmarker.task` model file is present
- Ensure you're in the virtual environment: `source bin/activate`
- Check all dependencies are installed: `pip list`

## PERFORMANCE TIPS

- **Lighting** - Good, even lighting improves detection accuracy
- **Background** - Plain backgrounds work better than cluttered ones
- **Hand Size** - Keep hand at a comfortable distance (1-2 feet from camera)
- **Camera Quality** - Higher resolution cameras provide better tracking
- **CPU Usage** - Close other applications if tracking is slow

## ADVANCED CUSTOMIZATION

### Change Tracking Point

By default, the cursor tracks landmark 5 (index finger first knuckle). To track a different point, modify:

```python
normalized_x, normalized_y = landmark_5.x, landmark_5.y
```

Other useful landmarks:
- `landmark_8` - Index finger tip
- `landmark_9` - Middle finger tip
- `landmark_0` - Wrist

### Add Custom Gestures

Add new gesture detection in the ratios section:

```python
elif ratios["T->M / PL"] < 0.15 and ratios["T->I / PL"] > 0.20:
    # Two-finger pinch detected
    pyautogui.doubleClick()
```

### Adjust Kalman Filter

For different smoothing characteristics, modify the Kalman filter parameters:

```python
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Lower = smoother, higher = more responsive
```

## NOTES

- **Hand Detection Release** - When hand leaves camera view, cursor automatically releases to screen center
- **Physical Mouse** - Can be used anytime by removing hand from view or pressing 'T' to disable
- **Gesture Cooldowns** - 0.4 second delay between same gesture triggers prevents accidental actions
- **Dual Orientation** - Supports both traditional webcam and overhead desk-mounted camera setups
- **Vulcan Salute Scroll** - Distinctive 🖖 gesture prevents accidental scroll activation
- **Visual Feedback** - Hand landmarks and on-screen text provide immediate feedback on gesture state
- PyAutoGUI failsafe is **disabled** in this script for smoother operation
- Cursor movement runs in a separate thread for better performance
- Frame counter is used for video mode processing in MediaPipe
- Distance ratios are normalized by palm length for hand size independence
- Scroll uses Vulcan salute detection to avoid conflicts with pinch gestures

## CREDITS

Based on MediaPipe Hand Landmarker by Google
