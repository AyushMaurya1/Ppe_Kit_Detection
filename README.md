**PPE Detection System**

This project implements a PPE Detection System, Human Detection using YOLOv3, and Face Recognition. These systems can analyze images, videos, or webcam feeds to detect compliance with safety requirements, human presence, and recognize faces. Alerts are triggered when violations are detected, such as non-compliance with PPE or an unknown face.


**Features**

Image Detection: Analyze images for PPE compliance.
Video Detection: Process video files to detect non-compliance.
Webcam Detection: Monitor a live webcam feed for PPE violations.
Audio Alert: Plays a buzzer sound when non-compliance is detected.
Real-Time Display: Annotated outputs with detected objects are displayed.


**Prerequisites**

Python: Ensure Python 3.8+ is installed.
Dependencies: Install the required libraries using:
bash
Copy code
pip install ultralytics opencv-python numpy pygame
pip install opencv-python-headless  # For YOLOv3 Human Detection


**YOLOv8 Models:**

PPE Detection Model: A YOLO model trained for PPE detection (yolo.pt).
Worker Detection Model: A YOLO model trained for detecting workers (person.pt).


**Buzzer Audio:**

Place the buzzer sound file (buzzer.wav) in the specified directory.

**Usage**

Run the Script:
python ppe_detection.py

Select Model:
PPE: For detecting PPE (e.g., helmets, vests, gloves).
Worker: For detecting workers.

Select Source:
Image: Provide the path to an image file.
Video: Provide the path to a video file.
Webcam: Use the default webcam for real-time monitoring.

**Key Functions**

1. Buzzer Initialization
Initializes the buzzer using the pygame library.
Triggers an alert when non-compliance is detected.

3. YOLO Model Loading
Dynamically loads the selected YOLO model.

5. Image Detection
Detects PPE violations in a static image.
Displays an annotated image with detection results.

7. Video Detection
Processes video frames to detect PPE violations.
Plays the buzzer sound asynchronously when non-compliance is detected.

9. Webcam Detection
Monitors a live feed from the webcam for PPE compliance.


**Configuration**

Modify the neceassry paths in the scripts as per your system.


**License**
This project is licensed under the MIT License - see the LICENSE file for details.
