# Hand Gesture Drawing App

This project implements a real-time hand gesture drawing application using OpenCV, MediaPipe, and Python. The application allows users to draw on a virtual canvas by using their hand gestures detected through a webcam.

## Features
- **Drawing with Index Finger**: When your index finger is raised, you can draw on the canvas.
- **Clear Canvas**: When all fingers are raised (gesture for all fingers open), the canvas gets cleared.
- **Change Color**: Use the 'A' and 'D' keys to change the color of the brush.
- **Change Brush Size**: Use the 'W' key to increase the brush size and 'S' to decrease it.
- **Real-time Drawing**: Draw freely with smooth brush strokes that follow the position of your index finger.

## Requirements
- Python 3.x
- OpenCV
- Mediapipe
- Numpy

## Installation

Install the required dependencies:

    ```
    pip install opencv-python mediapipe numpy
    ```

## How to Use

1. Open the terminal and run the main script:

    ```
    python main.py
    ```

2. The webcam feed will open up, and you can interact with the virtual canvas using hand gestures.
   - **Drawing**: Raise your index finger to start drawing.
   - **Clear Canvas**: Raise all your fingers to clear the canvas.
   - **Change Color**: Press 'A' to switch to the previous color and 'D' to switch to the next color.
   - **Change Brush Size**: Press 'W' to increase the brush size and 'S' to decrease it.

3. Press `ESC` to exit the application.

## Customization
- You can modify the colors, brush size limits, and detection/tracking confidence in the code to fit your preferences.
- The project uses a fixed canvas size (800x600), but this can be changed if desired.
  

## Acknowledgements
- This project utilizes **MediaPipe** for hand tracking, which provides real-time hand gesture recognition.
- OpenCV is used for capturing video input and drawing on the canvas.
