import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define colors and deque to store drawing points
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 255)]  # Red, Green, Blue, Yellow, White
color_names = ["Red", "Green", "Blue", "Yellow", "White"]  # Color names for display
color_index = 0
points = deque(maxlen=512)

# Create a black canvas
canvas = np.zeros((600, 800, 3), dtype=np.uint8)

# Moving average window size (smooths cursor movement)
window_size = 5
previous_positions = deque(maxlen=window_size)

# Default drawing size
drawing_size = 5
drawing_data = deque(maxlen=512)  # Store x, y, color, and size for each point


def detect_gesture(landmarks):
    if landmarks:
        fingers = []
        tips = [4, 8, 12, 16, 20]
        for i in range(1, 5):
            fingers.append(landmarks[tips[i]].y < landmarks[tips[i] - 2].y)

        if fingers == [True, False, False, False]:  # One finger up (index finger)
            return "DRAW"
        elif all(fingers):  # All fingers up
            return "CLEAR"
    return "NONE"


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    height, width, _ = frame.shape

    index_x, index_y = -1, -1  # Initialize default values for index_x and index_y

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            gesture = detect_gesture(landmarks)

            # Get the position of the index finger (tip)
            index_x = int(landmarks[8].x * width)
            index_y = int(landmarks[8].y * height)

            if gesture == "DRAW":
                # Add current position and size to the list for drawing (real-time tracking)
                drawing_data.append((index_x, index_y, color_index, drawing_size))
                # Add the current position to the list for smoothing
                previous_positions.append((index_x, index_y))
            elif gesture == "CLEAR":
                canvas.fill(0)
                drawing_data.clear()  # Clear the previous drawing data

    # Smooth the cursor position by averaging previous positions
    if len(previous_positions) > 0:
        smoothed_x = np.mean([pos[0] for pos in previous_positions]).astype(int)
        smoothed_y = np.mean([pos[1] for pos in previous_positions]).astype(int)

        # Add smoothed position to the points deque
        points.append((smoothed_x, smoothed_y, color_index))

    # Draw on canvas (drawing points and smoothing)
    for x, y, c, size in drawing_data:
        cv2.circle(canvas, (x, y), size, colors[c], -1)

    # Display the current color and instructions
    cv2.putText(frame, f"Current Color: {color_names[color_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                colors[color_index], 2, cv2.LINE_AA)
    cv2.putText(frame, "Use 'A' and 'D' to Change Color", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(frame, "Use 'W' and 'S' to Change Brush Size", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)

    # Show the frames
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Blackboard", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('a'):  # 'A' to change to the previous color
        color_index = (color_index - 1) % len(colors)
    elif key == ord('d'):  # 'D' to change to the next color
        color_index = (color_index + 1) % len(colors)
    elif key == ord('w'):  # 'W' to increase brush size
        drawing_size = min(drawing_size + 1, 20)  # Cap the max size
    elif key == ord('s'):  # 'S' to decrease brush size
        drawing_size = max(drawing_size - 1, 1)  # Cap the min size

cap.release()
cv2.destroyAllWindows()
