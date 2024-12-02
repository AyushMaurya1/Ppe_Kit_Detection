# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# from PIL import Image
# import matplotlib.pyplot as plt

# # Define paths for PPE and Worker models (change these as needed)
# PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\yolo.pt"
# WORKER_MODEL_PATH = "model\person.pt"

# # Define confidence threshold
# CONFIDENCE_THRESHOLD = 0.5

# # Load YOLO model
# def load_model(model_path):
#     return YOLO(model_path)

# # Detect and visualize PPE compliance in an image
# def detect_ppe_in_image(image_path, model, confidence):
#     try:
#         # Load the image
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

#         # Run model inference
#         results = model.predict(image, conf=confidence)
#         res_image = results[0].plot()  # Annotated image
#         boxes = results[0].boxes

#         # Display detection results
#         print(f"Detections in {image_path}:")
#         for box in boxes:
#             print(f"Class: {box.cls}, Confidence: {box.conf}, Coordinates: {box.xywh}")

#         # Display the annotated image
#         cv2.imshow("Detected PPE", cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in image: {e}")

# # Detect and visualize PPE compliance in a video
# def detect_ppe_in_video(video_path, model, confidence):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run model inference
#             results = model.predict(frame, conf=confidence)
#             res_frame = results[0].plot()  # Annotated frame

#             # Display the frame
#             cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))
            
#             # Break on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in video: {e}")

# # Main function
# if __name__ == "__main__":
#     # Choose model type
#     model_type = input("Select Model (PPE/Worker): ").strip().lower()
#     if model_type == "ppe":
#         model_path = PPE_MODEL_PATH
#     elif model_type == "worker":
#         model_path = WORKER_MODEL_PATH
#     else:
#         print("Invalid model type selected. Exiting.")
#         exit()

#     # Load selected model
#     model = load_model(model_path)

#     # Choose source type
#     source_type = input("Select Source (Image/Video): ").strip().lower()
#     if source_type == "image":
#         image_path = input("Enter path to the image file: ").strip()
#         if not Path(image_path).is_file():
#             print("Invalid image path. Exiting.")
#         else:
#             detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)

#     elif source_type == "video":
#         video_path = input("Enter path to the video file: ").strip()
#         if not Path(video_path).is_file():
#             print("Invalid video path. Exiting.")
#         else:
#             detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)

#     else:
#         print("Invalid source type selected. Exiting.")
























# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# from playsound import playsound  # For buzzer sound
# import pygame
# import os

# # Define paths for PPE and Worker models (change these as needed)
# PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\yolo.pt"
# WORKER_MODEL_PATH = "model\person.pt"

# # Define buzzer audio path (change to your file's location)
# BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.mp3"

# # Define confidence threshold
# CONFIDENCE_THRESHOLD = 0.5

# # Load YOLO model
# def load_model(model_path):
#     return YOLO(model_path)

# # Play buzzer sound
# # def play_buzzer():
# #     if os.path.exists(BUZZER_AUDIO_PATH):
# #         playsound(BUZZER_AUDIO_PATH)
# #     else:
# #         print("Buzzer audio file not found!")

# def play_buzzer():
#     if os.path.exists(BUZZER_AUDIO_PATH):
#         pygame.mixer.init()
#         pygame.mixer.music.load(BUZZER_AUDIO_PATH)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():  # Wait for the sound to finish
#             continue
#     else:
#         print("Buzzer audio file not found!")

# # Detect and visualize PPE compliance in an image
# def detect_ppe_in_image(image_path, model, confidence):
#     try:
#         # Load the image
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

#         # Run model inference
#         results = model.predict(image, conf=confidence)
#         res_image = results[0].plot()  # Annotated image
#         boxes = results[0].boxes

#         # Check for specific detections
#         buzzer_triggered = False
#         print(f"Detections in {image_path}:")
#         for box in boxes:
#             class_label = box.cls
#             confidence = box.conf
#             coordinates = box.xywh
#             print(f"Class: {class_label}, Confidence: {confidence}, Coordinates: {coordinates}")

#             # Trigger buzzer for specific classes
#             if class_label in ["no_vest", "no_hardhat", "no_gloves"]:
#                 buzzer_triggered = True

#         if buzzer_triggered:
#             print("Non-compliance detected! Playing buzzer...")
#             play_buzzer()

#         # Display the annotated image
#         cv2.imshow("Detected PPE", cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in image: {e}")

# # Detect and visualize PPE compliance in a video
# def detect_ppe_in_video(video_path, model, confidence):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run model inference
#             results = model.predict(frame, conf=confidence)
#             res_frame = results[0].plot()  # Annotated frame
#             boxes = results[0].boxes

#             # Check for specific detections
#             buzzer_triggered = False
#             for box in boxes:
#                 class_label = box.cls
#                 if class_label in ["no_vest", "no_hardhat", "no_gloves"]:
#                     buzzer_triggered = True

#             if buzzer_triggered:
#                 print("Non-compliance detected in video frame! Playing buzzer...")
#                 play_buzzer()

#             # Display the frame
#             cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))
            
#             # Break on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in video: {e}")

# # Main function
# if __name__ == "__main__":
#     # Choose model type
#     model_type = input("Select Model (PPE/Worker): ").strip().lower()
#     if model_type == "ppe":
#         model_path = PPE_MODEL_PATH
#     elif model_type == "worker":
#         model_path = WORKER_MODEL_PATH
#     else:
#         print("Invalid model type selected. Exiting.")
#         exit()

#     # Load selected model
#     model = load_model(model_path)

#     # Choose source type
#     source_type = input("Select Source (Image/Video): ").strip().lower()
#     if source_type == "image":
#         image_path = input("Enter path to the image file: ").strip()
#         if not Path(image_path).is_file():
#             print("Invalid image path. Exiting.")
#         else:
#             detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)

#     elif source_type == "video":
#         video_path = input("Enter path to the video file: ").strip()
#         if not Path(video_path).is_file():
#             print("Invalid video path. Exiting.")
#         else:
#             detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)

#     else:
#         print("Invalid source type selected. Exiting.")


# from playsound import playsound
# playsound(r"C:\Users\ayush\Downloads\buzzer.wav")

# import os
# BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.mp3"
# print(os.path.exists(BUZZER_AUDIO_PATH))  # Should print True
# import pygame

# def play_buzzer():
#     if os.path.exists(BUZZER_AUDIO_PATH):
#         pygame.mixer.init()
#         pygame.mixer.music.load(BUZZER_AUDIO_PATH)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():  # Wait for the sound to finish
#             continue
#     else:
#         print("Buzzer audio file not found!")


# import pygame
# import os

# BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.wav"

# def play_buzzer():
#     if os.path.exists(BUZZER_AUDIO_PATH):
#         pygame.mixer.init()
#         pygame.mixer.music.load(BUZZER_AUDIO_PATH)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             continue  # Wait for the sound to finish playing
#     else:
#         print("Audio file not found!")

# play_buzzer()


# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import pygame  # For buzzer sound
# import os

# # Define paths for PPE and Worker models (change these as needed)
# PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\yolo.pt"
# WORKER_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\person.pt"

# # Define buzzer audio path
# BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.wav"

# # Define confidence threshold
# CONFIDENCE_THRESHOLD = 0.5

# # Load YOLO model
# def load_model(model_path):
#     return YOLO(model_path)

# # Play buzzer sound
# def play_buzzer():
#     if os.path.exists(BUZZER_AUDIO_PATH):
#         pygame.mixer.init()
#         pygame.mixer.music.load(BUZZER_AUDIO_PATH)
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             continue  # Wait for the sound to finish playing
#     else:
#         print("Audio file not found!")

# # Detect and visualize PPE compliance in an image
# def detect_ppe_in_image(image_path, model, confidence):
#     try:
#         # Load the image
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

#         # Run model inference
#         results = model.predict(image, conf=confidence)
#         res_image = results[0].plot()  # Annotated image
#         boxes = results[0].boxes

#         # Check for specific detections
#         buzzer_triggered = False
#         print(f"Detections in {image_path}:")
#         for box in boxes:

#             class_label = box.cls
#             print(f"Detected class: {class_label}")
#             confidence = box.conf
#             coordinates = box.xywh
#             print(f"Class: {class_label}, Confidence: {confidence}, Coordinates: {coordinates}")

#             # Trigger buzzer for specific classes
#             if class_label in ["no_vest", "no_hardhat", "no_gloves"]:
#                 buzzer_triggered = True

#         if buzzer_triggered:
#             print("Non-compliance detected! Playing buzzer...")
#             play_buzzer()

#         # Display the annotated image
#         cv2.imshow("Detected PPE", cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in image: {e}")

# # Detect and visualize PPE compliance in a video
# def detect_ppe_in_video(video_path, model, confidence):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Run model inference
#             results = model.predict(frame, conf=confidence)
#             res_frame = results[0].plot()  # Annotated frame
#             boxes = results[0].boxes

#             # Check for specific detections
#             buzzer_triggered = False
#             class_names = model.names
#             for box in boxes:
#                 class_index = int(box.cls)
#                 class_label = class_names[class_index]
#                 print(f"Detected class: {class_label}")
#                 if class_label in ["no_vest", "no_hardhat", "no_gloves"]:
#                     buzzer_triggered = True
#             # for box in boxes:

#             #     class_label = box.cls
#             #     print(f"Detected class: {class_label}")
#             #     confidence = box.conf
#             #     coordinates = box.xywh
#             #     print(f"Class: {class_label}, Confidence: {confidence}, Coordinates: {coordinates}")

#             # Trigger buzzer for specific classes
#             if class_label in ["no_vest", "no_hardhat", "no_gloves"]:
#                 buzzer_triggered = True


#             if buzzer_triggered:
#                 print("Non-compliance detected in video frame! Playing buzzer...")
#                 play_buzzer()

#             # Display the frame
#             cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))
            
#             # Break on 'q' key press
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in video: {e}")

# # Main function
# if __name__ == "__main__":
#     # Initialize pygame mixer
#     pygame.init()

#     # Choose model type
#     model_type = input("Select Model (PPE/Worker): ").strip().lower()
#     if model_type == "ppe":
#         model_path = PPE_MODEL_PATH
#     elif model_type == "worker":
#         model_path = WORKER_MODEL_PATH
#     else:
#         print("Invalid model type selected. Exiting.")
#         exit()

#     # Load selected model
#     model = load_model(model_path)

#     # Choose source type
#     source_type = input("Select Source (Image/Video): ").strip().lower()
#     if source_type == "image":
#         image_path = input("Enter path to the image file: ").strip()
#         if not Path(image_path).is_file():
#             print("Invalid image path. Exiting.")
#         else:
#             detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)

#     elif source_type == "video":
#         video_path = input("Enter path to the video file: ").strip()
#         if not Path(video_path).is_file():
#             print("Invalid video path. Exiting.")
#         else:
#             detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)

#     else:
#         print("Invalid source type selected. Exiting.")



































# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import pygame  # For buzzer sound
# import os

# # Define paths for models and buzzer audio
# PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\yolo.pt"
# WORKER_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\person.pt"
# BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.wav"

# # Confidence threshold
# CONFIDENCE_THRESHOLD = 0.5

# # Initialize buzzer
# def initialize_buzzer():
#     if os.path.exists(BUZZER_AUDIO_PATH):
#         pygame.mixer.init()
#         pygame.mixer.music.load(BUZZER_AUDIO_PATH)
#     else:
#         print("Buzzer audio file not found!")

# # Play buzzer sound
# def play_buzzer():
#     if pygame.mixer.get_init():
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             pygame.time.delay(10)  # Non-blocking wait
#     else:
#         print("Buzzer not initialized. Skipping buzzer sound.")

# # Load YOLO model
# def load_model(model_path):
#     try:
#         return YOLO(model_path)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         exit()

# # Detect PPE compliance in an image
# def detect_ppe_in_image(image_path, model, confidence):
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError("Unable to load image.")
#         results = model.predict(image, conf=confidence)
#         res_image = results[0].plot()  # Annotated image

#         # Check detections and trigger buzzer
#         buzzer_triggered = any(
#             model.names[int(box.cls)] in ["no_vest", "no_hardhat", "no_gloves"]
#             for box in results[0].boxes
#         )
#         if buzzer_triggered:
#             print("Non-compliance detected! Playing buzzer...")
#             play_buzzer()

#         # Display result
#         cv2.imshow("Detected PPE", res_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in image: {e}")

# # Detect PPE compliance in a video
# # def detect_ppe_in_video(video_path, model, confidence):
# #     try:
# #         cap = cv2.VideoCapture(video_path)
# #         if not cap.isOpened():
# #             raise ValueError("Unable to open video file.")

# #         while cap.isOpened():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 break

# #             results = model.predict(frame, conf=confidence)
# #             res_frame = results[0].plot()  # Annotated frame

# #             # Check detections and trigger buzzer
# #             buzzer_triggered = any(
# #                 model.names[int(box.cls)] in ["no_vest", "no_hardhat", "no_gloves"]
# #                 for box in results[0].boxes
# #             )
# #             if buzzer_triggered:
# #                 print("Non-compliance detected in video frame! Playing buzzer...")
# #                 play_buzzer()

# #             # Display frame
# #             cv2.imshow("Detected PPE in Video", res_frame)

# #             # Exit on 'q' key
# #             if cv2.waitKey(1) & 0xFF == ord("q"):
# #                 break

# #         cap.release()
# #         cv2.destroyAllWindows()
# #     except Exception as e:
# #         print(f"Error detecting PPE in video: {e}")

# import time
# import threading

# # Function to play buzzer sound in a separate thread
# def play_buzzer_async():
#     if pygame.mixer.get_init():
#         pygame.mixer.music.play()
#     else:
#         print("Buzzer not initialized. Skipping buzzer sound.")

# def detect_ppe_in_video(video_path, model, confidence):
#     try:
#         print(model.names)
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("Unable to open video file.")

#         # Get the video frame rate (FPS)
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break  # End of video

#             # Run model inference
#             results = model.predict(frame, conf=confidence, verbose=False)
#             res_frame = results[0].plot()  # Annotated frame (bounding boxes)

#             # Check detections and trigger buzzer if needed
#             buzzer_triggered = any(
#                 model.names[int(box.cls)] in ["no_vest", "no_hardhat", "no_gloves","no_boots"]
#                 for box in results[0].boxes
#             )
#             if buzzer_triggered:
#                 print("Non-compliance detected in video frame! Playing buzzer...")
#                 # Play buzzer asynchronously in a new thread
#                 threading.Thread(target=play_buzzer_async).start()

#             # Display the frame
#             cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))

#             # Control the frame rate to match the video FPS
#             if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
#                 break  # Exit on 'q' key press

#         cap.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in video: {e}")


# # Main function
# def main():
#     pygame.init()  # Initialize pygame for buzzer
#     initialize_buzzer()

#     # Select model
#     model_type = input("Select Model (PPE/Worker): ").strip().lower()
#     if model_type == "ppe":
#         model_path = PPE_MODEL_PATH
#     elif model_type == "worker":
#         model_path = WORKER_MODEL_PATH
#     else:
#         print("Invalid model type selected. Exiting.")
#         return

#     model = load_model(model_path)

#     # Select source
#     source_type = input("Select Source (Image/Video): ").strip().lower()
#     if source_type == "image":
#         image_path = input("Enter path to the image file: ").strip()
#         if Path(image_path).is_file():
#             detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)
#         else:
#             print("Invalid image path. Exiting.")

#     elif source_type == "video":
#         video_path = input("Enter path to the video file: ").strip()
#         if Path(video_path).is_file():
#             detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)
#         else:
#             print("Invalid video path. Exiting.")

#     else:
#         print("Invalid source type selected. Exiting.")

#     pygame.quit()  # Cleanup pygame

# if __name__ == "__main__":
#     main()





# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import pygame  # For buzzer sound
# import os
# import time
# import threading

# # Define paths for models and buzzer audio
# PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\yolo.pt"
# WORKER_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\person.pt"
# BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.wav"

# # Confidence threshold
# CONFIDENCE_THRESHOLD = 0.5

# # Initialize buzzer
# def initialize_buzzer():
#     if os.path.exists(BUZZER_AUDIO_PATH):
#         pygame.mixer.init()
#         pygame.mixer.music.load(BUZZER_AUDIO_PATH)
#     else:
#         print("Buzzer audio file not found!")

# # Play buzzer sound
# def play_buzzer():
#     if pygame.mixer.get_init():
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             pygame.time.delay(10)  # Non-blocking wait
#     else:
#         print("Buzzer not initialized. Skipping buzzer sound.")

# # Load YOLO model
# def load_model(model_path):
#     try:
#         return YOLO(model_path)
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         exit()

# # Detect PPE compliance in an image
# def detect_ppe_in_image(image_path, model, confidence):
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError("Unable to load image.")
#         results = model.predict(image, conf=confidence)
#         res_image = results[0].plot()  # Annotated image

#         # Check detections and trigger buzzer
#         buzzer_triggered = any(
#             model.names[int(box.cls)] in ["no_vest", "no_hardhat", "no_gloves"]
#             for box in results[0].boxes
#         )
#         if buzzer_triggered:
#             print("Non-compliance detected! Playing buzzer...")
#             play_buzzer()

#         # Display result
#         cv2.imshow("Detected PPE", res_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in image: {e}")

# # Function to play buzzer sound in a separate thread
# def play_buzzer_async():
#     if pygame.mixer.get_init():
#         pygame.mixer.music.play()
#     else:
#         print("Buzzer not initialized. Skipping buzzer sound.")

# # Detect PPE compliance in a video
# def detect_ppe_in_video(video_path, model, confidence):
#     try:
#         print(model.names)
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("Unable to open video file.")

#         # Get the video frame rate (FPS)
#         fps = cap.get(cv2.CAP_PROP_FPS)

#         # Create the OpenCV window and make it stay on top
#         cv2.namedWindow("Detected PPE in Video", cv2.WINDOW_NORMAL)
#         cv2.setWindowProperty("Detected PPE in Video", cv2.WND_PROP_TOPMOST, 1)

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break  # End of video

#             # Run model inference
#             results = model.predict(frame, conf=confidence, verbose=False)
#             res_frame = results[0].plot()  # Annotated frame (bounding boxes)

#             # Check detections and trigger buzzer if needed
#             buzzer_triggered = any(
#                 model.names[int(box.cls)] in ["no_vest", "no_hardhat", "no_gloves", "no_boots"]
#                 for box in results[0].boxes
#             )
#             if buzzer_triggered:
#                 print("Non-compliance detected in video frame! Playing buzzer...")
#                 # Play buzzer asynchronously in a new thread
#                 threading.Thread(target=play_buzzer_async).start()

#             # Display the frame
#             cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))

#             # Control the frame rate to match the video FPS
#             if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
#                 break  # Exit on 'q' key press

#         cap.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"Error detecting PPE in video: {e}")

# # Main function
# def main():
#     pygame.init()  # Initialize pygame for buzzer
#     initialize_buzzer()

#     # Select model
#     model_type = input("Select Model (PPE/Worker): ").strip().lower()
#     if model_type == "ppe":
#         model_path = PPE_MODEL_PATH
#     elif model_type == "worker":
#         model_path = WORKER_MODEL_PATH
#     else:
#         print("Invalid model type selected. Exiting.")
#         return

#     model = load_model(model_path)

#     # Select source
#     source_type = input("Select Source (Image/Video): ").strip().lower()
#     if source_type == "image":
#         image_path = input("Enter path to the image file: ").strip()
#         if Path(image_path).is_file():
#             detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)
#         else:
#             print("Invalid image path. Exiting.")

#     elif source_type == "video":
#         video_path = input("Enter path to the video file: ").strip()
#         if Path(video_path).is_file():
#             detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)
#         else:
#             print("Invalid video path. Exiting.")

#     else:
#         print("Invalid source type selected. Exiting.")

#     pygame.quit()  # Cleanup pygame

# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# from ultralytics import YOLO
# from pathlib import Path
# import pygame  # For buzzer sound
# import os
# import time
# import threading
# import logging

# # Define paths for models and buzzer audio
# PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\yolo.pt"
# WORKER_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\person.pt"
# BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.wav"

# # Confidence threshold
# CONFIDENCE_THRESHOLD = 0.5

# # Set up logging for better debugging
# logging.basicConfig(level=logging.INFO)

# # Initialize buzzer
# def initialize_buzzer():
#     if os.path.exists(BUZZER_AUDIO_PATH):
#         pygame.mixer.init()
#         pygame.mixer.music.load(BUZZER_AUDIO_PATH)
#         logging.info("Buzzer audio loaded successfully.")
#     else:
#         logging.error("Buzzer audio file not found!")

# # Play buzzer sound
# def play_buzzer():
#     if pygame.mixer.get_init():
#         pygame.mixer.music.play()
#         while pygame.mixer.music.get_busy():
#             pygame.time.delay(10)  # Non-blocking wait
#     else:
#         logging.error("Buzzer not initialized. Skipping buzzer sound.")

# # Load YOLO model
# def load_model(model_path):
#     try:
#         logging.info(f"Loading model from {model_path}...")
#         model = YOLO(model_path)
#         logging.info(f"Model loaded successfully.")
#         return model
#     except Exception as e:
#         logging.error(f"Error loading model: {e}")
#         exit()

# # Detect PPE compliance in an image
# def detect_ppe_in_image(image_path, model, confidence):
#     try:
#         logging.info(f"Processing image: {image_path}")
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError("Unable to load image.")
        
#         results = model.predict(image, conf=confidence)
#         res_image = results[0].plot()  # Annotated image

#         # Check detections and trigger buzzer
#         buzzer_triggered = False
#         for box in results[0].boxes:
#             class_name = model.names[int(box.cls)]
#             confidence_score = box.conf
#             logging.info(f"Detected: {class_name} with confidence {confidence_score:.2f}")
#             if class_name in ["no_vest", "no_hardhat", "no_gloves"]:
#                 buzzer_triggered = True

#         if buzzer_triggered:
#             logging.warning("Non-compliance detected! Playing buzzer...")
#             play_buzzer()

#         # Display result
#         cv2.imshow("Detected PPE", res_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except Exception as e:
#         logging.error(f"Error detecting PPE in image: {e}")

# # Function to play buzzer sound in a separate thread
# def play_buzzer_async():
#     if pygame.mixer.get_init():
#         pygame.mixer.music.play()
#     else:
#         logging.error("Buzzer not initialized. Skipping buzzer sound.")

# # Detect PPE compliance in a video
# def detect_ppe_in_video(video_path, model, confidence):
#     try:
#         print(model.names)
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError("Unable to open video file.")

#         # Get the video frame rate (FPS)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         print(f"INFO:root:Video FPS: {fps}")  # More detailed info

#         # Create the OpenCV window and make it stay on top
#         cv2.namedWindow("Detected PPE in Video", cv2.WINDOW_NORMAL)
#         cv2.setWindowProperty("Detected PPE in Video", cv2.WND_PROP_TOPMOST, 1)

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break  # End of video

#             # Run model inference
#             results = model.predict(frame, conf=confidence, verbose=False)
            
#             # Inspect the results for debugging
#             # print(f"INFO:root:Results: {results}")
            
#             # Annotated frame (bounding boxes)
#             res_frame = results[0].plot()

#             # Check detections and trigger buzzer if needed
#             buzzer_triggered = any(
#                 model.names[int(box.cls)] in ["no_vest", "no_hardhat", "no_gloves", "no_boots"]
#                 for box in results[0].boxes
#             )
#             if buzzer_triggered:
#                 print("Non-compliance detected in video frame! Playing buzzer...")
#                 threading.Thread(target=play_buzzer_async).start()

#             # Display the frame
#             cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))

#             # Control the frame rate to match the video FPS
#             if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
#                 break  # Exit on 'q' key press

#         cap.release()
#         cv2.destroyAllWindows()
#     except Exception as e:
#         print(f"ERROR:root:Error detecting PPE in video: {e}")


# # Main function
# def main():
#     pygame.init()  # Initialize pygame for buzzer
#     initialize_buzzer()

#     # Select model
#     model_type = input("Select Model (PPE/Worker): ").strip().lower()
#     if model_type == "ppe":
#         model_path = PPE_MODEL_PATH
#     elif model_type == "worker":
#         model_path = WORKER_MODEL_PATH
#     else:
#         logging.error("Invalid model type selected. Exiting.")
#         return

#     model = load_model(model_path)

#     # Select source
#     source_type = input("Select Source (Image/Video): ").strip().lower()
#     if source_type == "image":
#         image_path = input("Enter path to the image file: ").strip()
#         if Path(image_path).is_file():
#             detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)
#         else:
#             logging.error("Invalid image path. Exiting.")

#     elif source_type == "video":
#         video_path = input("Enter path to the video file: ").strip()
#         if Path(video_path).is_file():
#             detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)
#         else:
#             logging.error("Invalid video path. Exiting.")

#     else:
#         logging.error("Invalid source type selected. Exiting.")

#     pygame.quit()  # Cleanup pygame

# if __name__ == "__main__":
#     main()








import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pygame  # For buzzer sound
import os
import time
import threading
import logging

# Define paths for models and buzzer audio
PPE_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\yolo.pt"
WORKER_MODEL_PATH = r"C:\Users\ayush\Downloads\firstmilestone1\firstmilestone\model\person.pt"
BUZZER_AUDIO_PATH = r"C:\Users\ayush\Downloads\buzzer.wav"

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)

# Initialize buzzer
def initialize_buzzer():
    if os.path.exists(BUZZER_AUDIO_PATH):
        pygame.mixer.init()
        pygame.mixer.music.load(BUZZER_AUDIO_PATH)
        logging.info("Buzzer audio loaded successfully.")
    else:
        logging.error("Buzzer audio file not found!")

# Play buzzer sound
def play_buzzer():
    if pygame.mixer.get_init():
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.delay(10)  # Non-blocking wait
    else:
        logging.error("Buzzer not initialized. Skipping buzzer sound.")

# Load YOLO model
def load_model(model_path):
    try:
        logging.info(f"Loading model from {model_path}...")
        model = YOLO(model_path)
        logging.info(f"Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

# Detect PPE compliance in an image
def detect_ppe_in_image(image_path, model, confidence):
    try:
        logging.info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Unable to load image.")
        
        results = model.predict(image, conf=confidence)
        res_image = results[0].plot()  # Annotated image

        # Check detections and trigger buzzer
        buzzer_triggered = False
        detected_items = []

        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            confidence_score = float(box.conf)  # Ensure this is a float
            detected_items.append(f"{class_name} (Confidence: {confidence_score:.2f})")
            # logging.info(f"Detected: {class_name} with confidence {confidence_score:.2f}")

            if class_name in ["no_vest", "no_hardhat", "no_gloves"]:
                buzzer_triggered = True

        # Print the detected items in the terminal
        print("\nDetected Items:")
        for item in detected_items:
            print(item)

        if buzzer_triggered:
            logging.warning("Non-compliance detected! Playing buzzer...")
            play_buzzer()

        # Display result
        cv2.imshow("Detected PPE", res_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error detecting PPE in image: {e}")



# Function to play buzzer sound in a separate thread
def play_buzzer_async():
    if pygame.mixer.get_init():
        pygame.mixer.music.play()
    else:
        logging.error("Buzzer not initialized. Skipping buzzer sound.")

# Detect PPE compliance in a video
def detect_ppe_in_video(video_path, model, confidence):
    try:
        print(model.names)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file.")

        # Get the video frame rate (FPS)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"INFO:root:Video FPS: {fps}")  # More detailed info

        # Create the OpenCV window and make it stay on top
        cv2.namedWindow("Detected PPE in Video", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Detected PPE in Video", cv2.WND_PROP_TOPMOST, 1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Run model inference
            results = model.predict(frame, conf=confidence, verbose=False)
            
            # Annotated frame (bounding boxes)
            res_frame = results[0].plot()

            # Check detections and trigger buzzer if needed
            buzzer_triggered = False
            detected_items = []

            for box in results[0].boxes:
                class_name = model.names[int(box.cls)]
                confidence_score = float(box.conf)  # Ensure this is a float
                detected_items.append(f"{class_name} (Confidence: {confidence_score:.2f})")
                
                # Print detected items
                # print(f"Detected: {class_name} with confidence {confidence_score:.2f}")
                
                if class_name in ["no_vest", "no_hardhat", "no_gloves", "no_boots"]:
                    buzzer_triggered = True

            # Print the detected items in the terminal
            print("\nDetected Items in Current Frame:")
            for item in detected_items:
                print(item)

            if buzzer_triggered:
                print("\nNon-compliance detected in video frame! Playing buzzer...")
                threading.Thread(target=play_buzzer_async).start()

            # Display the frame
            cv2.imshow("Detected PPE in Video", cv2.cvtColor(res_frame, cv2.COLOR_RGB2BGR))

            # Control the frame rate to match the video FPS
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break  # Exit on 'q' key press

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"ERROR:root:Error detecting PPE in video: {e}")

# Main function
def main():
    pygame.init()  # Initialize pygame for buzzer
    initialize_buzzer()

    # Select model
    model_type = input("Select Model (PPE/Worker): ").strip().lower()
    if model_type == "ppe":
        model_path = PPE_MODEL_PATH
    elif model_type == "worker":
        model_path = WORKER_MODEL_PATH
    else:
        logging.error("Invalid model type selected. Exiting.")
        return

    model = load_model(model_path)

    # Select source
    source_type = input("Select Source (Image/Video): ").strip().lower()
    if source_type == "image":
        image_path = input("Enter path to the image file: ").strip()
        if Path(image_path).is_file():
            detect_ppe_in_image(image_path, model, CONFIDENCE_THRESHOLD)
        else:
            logging.error("Invalid image path. Exiting.")

    elif source_type == "video":
        video_path = input("Enter path to the video file: ").strip()
        if Path(video_path).is_file():
            detect_ppe_in_video(video_path, model, CONFIDENCE_THRESHOLD)
        else:
            logging.error("Invalid video path. Exiting.")

    else:
        logging.error("Invalid source type selected. Exiting.")

    pygame.quit()  # Cleanup pygame

if __name__ == "__main__":
    main()
