import cv2
import os
import pandas as pd
import datetime
from ultralytics import YOLO

# Initialize LBPH model
lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.read('lbph_model.yml')

# Initialize YOLO model for face detection
model = YOLO('yolov8n-face.pt')

# Attendance DataFrame with 'Name', 'ID', 'Date', 'Time'
col_names = ['Name', 'ID', 'Date', 'Time']
attendance = pd.DataFrame(columns=col_names)

# Load dataset ('ID', 'Name')
dataset = pd.read_csv("data.csv")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open the camera")
    exit()

cv2.namedWindow('Face Recognition Attendance System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Recognition Attendance System', 1100, 650)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame")
        break

    # Detect faces using YOLOv8
    results = model(frame)
    faces = results[0].boxes

    if faces is not None:
        for face in faces:
            x1, y1, x2, y2 = face.xyxy[0].int().tolist()
            
            # Check if face coordinates are valid
            if x1 >= 0 and y1 >= 0 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                face_img = frame[y1:y2, x1:x2]
                
                # Process only if face detection is large enough
                if face_img.shape[0] > 10 and face_img.shape[1] > 10:
                    try:
                        # Convert and resize face for LBPH
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        gray_face = cv2.resize(gray_face, (200, 200))
                        
                        # Predict the ID using LBPH
                        Id, confidence = lbph.predict(gray_face)
                        
                        # Set default values
                        name = "Unknown"
                        color = (0, 0, 255)  # Red for unknown faces
                        
                        # Check if ID exists in dataset and confidence is good
                        if confidence < 40 and not dataset.loc[dataset['ID'] == Id].empty:
                            name = dataset.loc[dataset['ID'] == Id, 'Name'].values[0]
                            color = (0, 255, 0)  # Green for recognized faces
                            
                            # Record attendance for recognized faces
                            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            attendance = pd.concat([
                                attendance,
                                pd.DataFrame([[name, Id, current_time.split()[0], current_time.split()[1]]],
                                           columns=col_names)
                            ], ignore_index=True)
                        
                        # Draw bounding box and text for all faces
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Add text background for better visibility
                        text = f"ID: {Id if name != 'Unknown' else 'N/A'} | {name}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0], y1), color, -1)
                        cv2.putText(frame, text, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        if name != "Unknown":
                            # Add confidence text for recognized faces
                            conf_text = f"Confidence: {100 - confidence:.2f}%"
                            cv2.putText(frame, conf_text, (x1, y2 + 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

    # Remove duplicates after detection loop
    attendance.drop_duplicates(subset=['ID'], keep='first', inplace=True)
    
    # Display the frame
    cv2.imshow('Face Recognition Attendance System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save attendance to CSV
os.makedirs("Attendance", exist_ok=True)
file_path = f"Attendance/Attendance_{datetime.datetime.now().strftime('%Y-%m-%d')}.csv"

if os.path.exists(file_path):
    existing_data = pd.read_csv(file_path)
    attendance = pd.concat([existing_data, attendance]).drop_duplicates(subset=['ID'], keep='first')

attendance.to_csv(file_path, index=False)
print(f"Attendance saved to {file_path}")

cap.release()
cv2.destroyAllWindows()







































# import cv2
# import os
# import pandas as pd
# import datetime

# # Initialize LBPH model
# lbph = cv2.face.LBPHFaceRecognizer_create()
# lbph.read('lbph_model.yml')  

# # Initializing YuNet 
# yunet = cv2.FaceDetectorYN.create(
#     model='face_detection_yunet_2023mar.onnx',
#     config='',
#     input_size=(320, 320),
#     score_threshold=0.9,
#     nms_threshold=0.3,
#     top_k=5000
# )

# # attendance DataFrame with 'Name', 'ID', 'Date', 'Time'
# col_names = ['Name', 'ID', 'Date', 'Time']
# attendance = pd.DataFrame(columns=col_names)

# # CSV has ('ID', 'Name')
# dataset = pd.read_csv("data.csv")  

# # Start webcam
# cap = cv2.VideoCapture(1)
# if not cap.isOpened():
#     print("Cannot open the camera")
#     exit()

# # window and set its size
# cv2.namedWindow('Face Recognition Attendance System', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Face Recognition Attendance System', 1100, 650)

# serial_number = 1  

# # Main loop
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab a frame")
#         break

#     # Resize frame for YuNet
#     height, width = frame.shape[:2]
#     yunet.setInputSize((width, height))
#     _, faces = yunet.detect(frame)

#     if faces is not None:
#         for face in faces:
#             x, y, w, h = face[:4].astype(int)
            
#             # Ensure the bounding box is within the frame dimensions
#             if x >= 0 and y >= 0 and x + w <= width and y + h <= height:
#                 face_img = frame[y:y + h, x:x + w]
#                 gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

#                 # Predict the ID using LBPH
#                 Id, confidence = lbph.predict(gray_face)

#                 if not dataset.loc[dataset['ID'] == Id].empty:
#                     name = dataset.loc[dataset['ID'] == Id, 'Name'].values[0]
#                 else:
#                     name = "Unknown"


#                 if confidence < 60:  
#                     current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                     new_entry = pd.Series([name, Id, current_time.split()[0], current_time.split()[1]], 
#                                         index=attendance.columns)
#                     attendance = pd.concat([attendance, pd.DataFrame([new_entry])], ignore_index=True)

#                     serial_number += 1
#                     cv2.putText(frame, f"ID: {Id} || Name: {name}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#                     accuracy = 100 - confidence
#                     accuracy_text = f"Accuracy: {accuracy:.2f}%"
#                     cv2.putText(frame, accuracy_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)


#         # Removing duplicates from attendance (only one entry per ID)
#         attendance.drop_duplicates(subset=['ID'], keep='first', inplace=True)

#     cv2.imshow('Face Recognition Attendance System', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Save attendance to CSV
# attendance_dir = "Attendance"
# os.makedirs(attendance_dir, exist_ok=True)
# date = datetime.datetime.now().strftime('%Y-%m-%d')
# file_path = f"{attendance_dir}/Attendance_{date}.csv"

# # If the file exists, append the new data to it; otherwise, create a new file
# if os.path.exists(file_path):
#     existing_data = pd.read_csv(file_path)
#     attendance = pd.concat([existing_data, attendance]).drop_duplicates(subset=['ID'], keep='first')

# attendance.to_csv(file_path, index=False)
# print(f"Attendance saved to {file_path}")

# cap.release()
# cv2.destroyAllWindows()