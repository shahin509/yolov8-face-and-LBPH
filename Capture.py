# Deepseeek

import cv2
import os
import csv
from ultralytics import YOLO

# Load YOLOv8 face detection model
yolo_model = YOLO('yolov8n-face.pt')

# Directories and CSV file
output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)
csv_file = 'data.csv'

# Initialize or append to the CSV file
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Name'])

# User input
unique_id = input("Enter a unique ID: ").strip()
name = input("Enter the person's name: ").strip()

# Save to CSV
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([unique_id, name])

serial_number = 1
cap = cv2.VideoCapture(0)

# Optimize camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Capturing... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faster face detection with YOLO
    results = yolo_model(frame, imgsz=320, verbose=False)  # Reduced input size
    faces = results[0].boxes.xyxy.cpu().numpy()

    if len(faces) > 0:
        # Get largest face
        face = max(faces, key=lambda f: (f[2]-f[0])*(f[3]-f[1]))
        x, y, x2, y2 = map(int, face[:4])
        
        # Save face
        cv2.imwrite(
            os.path.join(output_dir, f"{name}.{unique_id}.{serial_number}.jpg"),
            frame[y:y2, x:x2]
        )
        print(f"Saved sample {serial_number}")
        serial_number += 1

        # Break after 50 samples
        if serial_number > 50:
            break

    # Display
    cv2.imshow('Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Capture completed successfully!")
































# # Claude

# import cv2
# import os
# import csv
# from ultralytics import YOLO
# import threading
# from queue import Queue
# import time

# class FaceCapture:
#     def __init__(self):
#         self.output_dir = 'captured_images'
#         self.csv_file = 'data.csv'
#         self.yolo_model = YOLO('yolov8n-face.pt')
#         self.image_queue = Queue(maxsize=10)
#         self.is_running = True
        
#         # Create output directory
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         # Initialize CSV if it doesn't exist
#         if not os.path.exists(self.csv_file):
#             with open(self.csv_file, mode='w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(['ID', 'Name'])

#     def save_image_worker(self):
#         while self.is_running:
#             if not self.image_queue.empty():
#                 face_img, filename = self.image_queue.get()
#                 if face_img is not None:
#                     img_path = os.path.join(self.output_dir, filename)
#                     cv2.imwrite(img_path, face_img)
#                     print(f"Saved: {img_path}")
#                 self.image_queue.task_done()
#             else:
#                 time.sleep(0.01)  # Short sleep to prevent CPU overload

#     def capture_faces(self):
#         # Get user input
#         unique_id = input("Enter a unique ID for the person: ").strip()
#         name = input("Enter the person's name: ").strip()

#         # Save user info to CSV
#         with open(self.csv_file, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([unique_id, name])

#         # Start the saving worker thread
#         save_thread = threading.Thread(target=self.save_image_worker)
#         save_thread.daemon = True
#         save_thread.start()

#         # Initialize camera
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("Cannot open the camera")
#             return

#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for speed
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
#         serial_number = 1
#         last_capture_time = time.time()
#         capture_interval = 0.1  # Capture every 100ms

#         print("Capturing images automatically... Press 'q' to stop.")

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 break

#             current_time = time.time()
            
#             # Only process frame if enough time has passed
#             if current_time - last_capture_time >= capture_interval:
#                 # Detect faces using YOLOv8
#                 results = self.yolo_model(frame, conf=0.5)  # Add confidence threshold
#                 faces = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []

#                 for face in faces:
#                     x, y, x2, y2 = map(int, face[:4])
#                     # Add padding to face crop
#                     pad = 20
#                     y = max(0, y - pad)
#                     y2 = min(frame.shape[0], y2 + pad)
#                     x = max(0, x - pad)
#                     x2 = min(frame.shape[1], x2 + pad)
                    
#                     face_img = frame[y:y2, x:x2].copy()
                    
#                     if face_img.size > 0:  # Check if face crop is valid
#                         filename = f"{name}.{unique_id}.{serial_number}.jpg"
#                         self.image_queue.put((face_img, filename))
#                         serial_number += 1

#                     # Draw rectangle on display frame
#                     cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

#                 last_capture_time = current_time

#             # Show live feed
#             cv2.imshow('Capture - Press q to quit', frame)

#             # Exit conditions
#             if cv2.waitKey(1) & 0xFF == ord('q') or serial_number > 50:
#                 break

#         # Cleanup
#         self.is_running = False
#         cap.release()
#         cv2.destroyAllWindows()
#         self.image_queue.join()  # Wait for remaining images to be saved

# if __name__ == "__main__":
#     face_capture = FaceCapture()
#     face_capture.capture_faces()






































# import cv2
# import os
# import time
# import csv

# # Initialize YuNet face detector
# yunet = cv2.FaceDetectorYN.create(
#     model='face_detection_yunet_2023mar.onnx',
#     config='',
#     input_size=(320, 320),
#     score_threshold=0.9,
#     nms_threshold=0.3,
#     top_k=5000
# )

# # Directories and CSV file
# output_dir = 'captured_images'
# os.makedirs(output_dir, exist_ok=True)
# csv_file = 'data.csv'

# # Initialize or append to the CSV file
# if not os.path.exists(csv_file):
#     with open(csv_file, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['ID', 'Name'])  # Write header

# # User input for name and ID
# unique_id = input("Enter a unique ID for the person: ").strip()
# name = input("Enter the person's name: ").strip()

# # Save user info to CSV
# with open(csv_file, mode='a', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow([unique_id, name])

# serial_number = 1  # Start serial number

# # Start webcam
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     print("Cannot open the camera")
#     exit()

# print("Capturing images automatically... Press 'q' to stop.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Detect faces
#     height, width = frame.shape[:2]
#     yunet.setInputSize((width, height))
#     _, faces = yunet.detect(frame)

#     if faces is not None:
#         for face in faces:
#             x, y, w, h = face[:4].astype(int)
#             face_img = frame[y:y + h, x:x + w]

#             # Save image
#             file_name = f"{name}.{unique_id}.{serial_number}.jpg"    #### name.id.sl_num
#             img_path = os.path.join(output_dir, file_name)
#             cv2.imwrite(img_path, face_img)
#             print(f"Saved: {img_path}")
#             serial_number += 1

#     # Show live feed
#     for face in faces if faces is not None else []:
#         x, y, w, h = face[:4].astype(int)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imshow('Capture - Press q to quit', frame)

#     # Exit conditions
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         print("Quitting...")
#         break
#     elif serial_number >= 50:
#         print("Collected 50 samples. Stopping...")
#         break

#     # Add delay to avoid overloading
#     time.sleep(0.5)

# cap.release()
# cv2.destroyAllWindows()