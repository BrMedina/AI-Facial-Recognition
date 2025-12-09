'''
PyPower Projects
Face Recognition
'''

#USAGE : python test.py

import cv2
import numpy as np
import os
import pickle

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
    print("Model loaded successfully!")
    print(f"Recognized people: {list(labels_dict.keys())}")
else:
    print("Model not found. Please train the model first.")
    print("Create folders: 'training_images/brother' and 'training_images/little_brother'")
    print("Add face images of each person in their respective folders.")
    print("Then run: python train_faces.py")
    exit()

cap = cv2.VideoCapture(0)

# If default backend fails (common on Windows), retry with DirectShow
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not access the camera. Ensure it is free and retry.")
    exit()



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

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(200,200),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            # Recognize face
            label, confidence = face_recognizer.predict(roi_gray)
            person_name = class_labels[label] if label < len(class_labels) else 'Unknown'
            
            # Display result
            confidence_text = f"{person_name} ({100-int(confidence)}%)"
            print(f"\nDetected: {confidence_text}")
            label_position = (x, y-10)
            cv2.putText(frame, confidence_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n")
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()