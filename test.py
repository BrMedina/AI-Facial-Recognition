'''
PyPower Projects
Face Recognition - UI Friendly Version
'''

#USAGE : python test.py

import cv2
import numpy as np
import os
import pickle
import ctypes

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Create recognizer (using local binary patterns histograms)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained model if exists
model_path = 'face_model.yml'
labels_path = 'labels.pkl'

if os.path.exists(model_path) and os.path.exists(labels_path):
    face_recognizer.read(model_path)
    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)
    class_labels = {v: k for k, v in labels_dict.items()}
    print("✓ Model loaded successfully!")
    print(f"✓ Recognized people: {', '.join(list(labels_dict.keys()))}")
else:
    print("✗ Model not found. Please train the model first.")
    print("Create folders: 'training_images/brother' and 'training_images/little_brother'")
    print("Add face images of each person in their respective folders.")
    print("Then run: python train_faces.py")
    exit()

# UI Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 1.2
FONT_THICKNESS = 2
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (0, 255, 0)
ACCENT_COLOR = (0, 200, 255)
RECT_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 165, 255)

cap = cv2.VideoCapture(0)

# Set camera resolution to 1280x720 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# If default backend fails (common on Windows), retry with DirectShow
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not access the camera. Ensure it is free and retry.")
    exit()

# Prepare window centering helper
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)
window_initialized = False

# Stats tracking
face_count = 0
detected_people = {}


def add_ui_panel(frame, face_count, detected_people):
    """Add info panel at top of frame"""
    h, w = frame.shape[:2]
    
    # Darken top area for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "Facial Recognition", (20, 35), FONT, 1.5, TEXT_COLOR, 2)
    
    # Stats
    stats_text = f"Faces Detected: {face_count} | Press Q to Exit"
    cv2.putText(frame, stats_text, (20, 70), FONT, 0.9, ACCENT_COLOR, 1)
    
    return frame


def add_detection_info(frame, person_name, confidence, x, y, w, h):
    """Add detection info with better styling"""
    # Draw rounded rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), RECT_COLOR, 3)
    
    # Add label background
    label_text = f"{person_name}"
    confidence_pct = 100 - int(confidence)
    confidence_text = f"{confidence_pct}% Match"
    
    text_size = cv2.getTextSize(label_text, FONT, FONT_SIZE, FONT_THICKNESS)[0]
    
    # Label background
    cv2.rectangle(frame, (x, y - 80), (x + text_size[0] + 50, y - 20), BG_COLOR, -1)
    cv2.rectangle(frame, (x, y - 80), (x + text_size[0] + 50, y - 20), TEXT_COLOR, 3)
    
    # Person name - larger
    cv2.putText(frame, label_text, (x + 15, y - 45), FONT, 1.5, TEXT_COLOR, 3)
    
    # Confidence - MUCH LARGER and more visible
    conf_size = cv2.getTextSize(confidence_text, FONT, 2.0, 3)[0]
    conf_x = max(x, (frame.shape[1] - conf_size[0]) // 2)
    
    # Draw confidence box at bottom center
    cv2.rectangle(frame, (conf_x - 20, frame.shape[0] - 80), 
                 (conf_x + conf_size[0] + 20, frame.shape[0] - 20), 
                 BG_COLOR, -1)
    cv2.rectangle(frame, (conf_x - 20, frame.shape[0] - 80), 
                 (conf_x + conf_size[0] + 20, frame.shape[0] - 20), 
                 ACCENT_COLOR, 3)
    
    cv2.putText(frame, confidence_text, (conf_x, frame.shape[0] - 35), 
               FONT, 2.0, ACCENT_COLOR, 3)
    
    return frame


while True:
    # Grab a single frame of video
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Warning: Failed to read from camera. Is it in use by another app?")
        break
    frame = cv2.flip(frame, 1)
    
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    face_count = len(faces)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(200,200),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            # Recognize face
            label, confidence = face_recognizer.predict(roi_gray)
            
            # If confidence is too low, label as "others"
            confidence_threshold = 60  # Adjust this value (lower = stricter)
            if confidence > confidence_threshold:
                person_name = "others"
            else:
                person_name = class_labels[label] if label < len(class_labels) else 'Unknown'
            
            # Track detected people
            if person_name not in detected_people:
                detected_people[person_name] = 0
            detected_people[person_name] += 1
            
            # Add detection info with UI
            frame = add_detection_info(frame, person_name, confidence, x, y, w, h)
            
            # Console output
            confidence_pct = 100 - int(confidence)
            print(f"✓ Detected: {person_name} ({confidence_pct}% confidence)")
        else:
            cv2.putText(frame,'No Face Data',(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),2)

    # Add UI panel at top
    frame = add_ui_panel(frame, face_count, detected_people)
    
    # Show no faces message if none detected
    if face_count == 0:
        h, w = frame.shape[:2]
        cv2.putText(frame, "No faces detected - Look at the camera", (w//2 - 300, h//2), 
                   FONT, 1.2, WARNING_COLOR, 2)

    if not window_initialized:
        cv2.namedWindow('Prototype', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Prototype', 1280, 720)
        cv2.moveWindow('Prototype', max(0, (screen_w - 1280)//2), max(0, (screen_h - 720)//2))
        window_initialized = True

    cv2.imshow('Prototype',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n✓ Program closed.")
        break

cap.release()
cv2.destroyAllWindows()