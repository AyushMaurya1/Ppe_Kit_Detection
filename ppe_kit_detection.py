# import cv2
# import numpy as np
# import os

# # Paths to the YOLO model files
# weights_path = "yolov3.weights"  # Pre-trained YOLO weights for helmet detection
# config_path = "yolov3.cfg"       # YOLO configuration file
# classes_file = "coco.names"  # File containing class names (e.g., 'helmet', 'no-helmet')

# # Load YOLO model
# net = cv2.dnn.readNet(weights_path, config_path)

# # Load class labels (helmet and no-helmet)
# classes = []
# with open(classes_file, "r") as file:
#     classes = [line.strip() for line in file.readlines()]

# # Get output layers
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Detection thresholds
# CONF_THRESHOLD = 0.5  # Minimum confidence to keep detection
# NMS_THRESHOLD = 0.4   # Non-Maximum Suppression threshold to remove overlapping boxes

# # Function to process a video
# def process_video(video_path, output_dir=None):
#     cap = cv2.VideoCapture(video_path)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     # Video writer for output
#     video_writer = None
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f"helmet_detection_{os.path.basename(video_path)}")
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         height, width, _ = frame.shape

#         # Prepare the frame for YOLO
#         blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#         net.setInput(blob)
#         outs = net.forward(output_layers)

#         # Lists to hold detection data
#         boxes, confidences, class_ids = [], [], []

#         # Parse detections
#         for out_layer in outs:
#             for detection in out_layer:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > CONF_THRESHOLD:
#                     # Bounding box coordinates
#                     center_x, center_y = int(detection[0] * width), int(detection[1] * height)
#                     w, h = int(detection[2] * width), int(detection[3] * height)
#                     x, y = int(center_x - w / 2), int(center_y - h / 2)

#                     boxes.append([x, y, w, h])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)

#         # Non-Maximum Suppression to remove overlapping boxes
#         indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

#         # Draw bounding boxes
#         for i in indexes.flatten():
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]
#             color = (0, 255, 0) if label == "helmet" else (0, 0, 255)  # Green for helmet, Red for no-helmet

#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # Show the frame
#         cv2.imshow("Helmet Detection", frame)

#         # Write the frame to output video
#         if video_writer:
#             video_writer.write(frame)

#         # Exit on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     if video_writer:
#         video_writer.release()
#     cv2.destroyAllWindows()

#     print(f"Finished processing {video_path}")

# # Main function
# def main():
#     video_folder = "videos"  # Input folder containing video files
#     output_dir = "output"    # Folder to save processed videos

#     # Get all video files in the folder
#     video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

#     if not video_files:
#         print("No videos found in the 'videos' directory.")
#         return

#     for video_file in video_files:
#         video_path = os.path.join(video_folder, video_file)
#         print(f"Processing {video_file}...")
#         process_video(video_path, output_dir)

# if __name__ == "__main__":
#     main()





import cv2
import numpy as np

# Load Haar Cascade classifier for helmet detection
helmet_cascade = cv2.CascadeClassifier('haarcascade_helmet.xml')  # Path to your helmet Haar Cascade

# Function to detect helmets in an image
def detect_helmet(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect helmets
    helmets = helmet_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))
    print(f"Detected {len(helmets)} helmets")  # Debugging line

    # Draw bounding boxes for helmets
    for (x, y, w, h) in helmets:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Helmet", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# Function to process an image file
def process_image(image_path):
    image = cv2.imread(image_path)
    processed_image = detect_helmet(image)
    cv2.imshow("Helmet Detection", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process a video file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detect_helmet(frame)
        cv2.imshow("Helmet Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to process live webcam feed
def process_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from webcam")
            break

        processed_frame = detect_helmet(frame)
        cv2.imshow("Helmet Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to handle user inputs
def main():
    input_type = input("Choose input type (image, video, webcam): ").strip().lower()

    if input_type == "image":
        image_path = input("Enter the path to the image file: ").strip()
        process_image(image_path)

    elif input_type == "video":
        video_path = input("Enter the path to the video file: ").strip()
        process_video(video_path)

    elif input_type == "webcam":
        print("Starting webcam...")
        process_webcam()

    else:
        print("Invalid input type. Please choose 'image', 'video', or 'webcam'.")

if __name__ == "__main__":
    main()




