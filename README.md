**PPE Detection System**
This project implements a PPE (Personal Protective Equipment) detection system using YOLOv8 models. The system can analyze images, videos, or webcam feeds to detect compliance with PPE requirements. When non-compliance is detected, it triggers a buzzer sound as an alert.

**Features**
Image Detection: Analyze images for PPE compliance.
Video Detection: Process video files to detect non-compliance.
Webcam Detection: Monitor a live webcam feed for PPE violations.
Audio Alert: Plays a buzzer sound when non-compliance is detected.
Real-Time Display: Annotated outputs with detected objects are displayed.

**Prerequisites**
Python: Ensure Python 3.8+ is installed.
Dependencies: Install the required Python libraries using:
pip install ultralytics opencv-python numpy pygame

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
2. YOLO Model Loading
Dynamically loads the selected YOLO model.
3. Image Detection
Detects PPE violations in a static image.
Displays an annotated image with detection results.
4. Video Detection
Processes video frames to detect PPE violations.
Plays the buzzer sound asynchronously when non-compliance is detected.
5. Webcam Detection
Monitors a live feed from the webcam for PPE compliance.

**Configuration**
Modify the following paths in the script as per your system:
PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\ppe_detection\model\yolo.pt"
WORKER_MODEL_PATH = r"C:\Users\ayush\Downloads\ppe_detection\model\person.pt"
BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.wav"

**Troubleshooting**
Model Not Loading:

Ensure the model files are present in the specified directory.
Verify the ultralytics package is installed.
Buzzer Not Playing:

Confirm buzzer.wav exists at the specified path.
Check if the pygame library is initialized properly.
Webcam Issues:

Ensure the webcam is connected and accessible.
Verify permissions for accessing the webcam.
Dependencies Not Found:

Run pip install -r requirements.txt with a requirements.txt file listing:
Copy code
ultralytics
opencv-python
numpy
pygame
Future Enhancements
Add support for additional detection classes (e.g., face shields).
Implement a logging system for compliance records.
Optimize real-time performance for high-resolution inputs.
License
This project is open-source and can be modified or distributed under the MIT license.
