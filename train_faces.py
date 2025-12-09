'''
Face Recognition Training Script
Run this first to train the model with your face images
'''

import cv2
import os
import numpy as np
import pickle

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

training_data = []
labels = []
label_dict = {}
label_count = 0

# Path to training images
training_dir = 'training_images'

if not os.path.exists(training_dir):
    print(f"Creating {training_dir} folder...")
    os.makedirs(training_dir)
    print("Please create subfolders for each person:")
    print("  - training_images/brother")
    print("  - training_images/little_brother")
    print("Add 5-10 images of each person's face to their folder.")
    print("Supported formats: jpg, png, jpeg")
    exit()

# Load images from subfolders
for person_name in os.listdir(training_dir):
    person_path = os.path.join(training_dir, person_name)
    
    if not os.path.isdir(person_path):
        continue
    
    if person_name not in label_dict:
        label_dict[person_name] = label_count
        label_count += 1
    
    print(f"Loading images for {person_name}...")
    
    for image_name in os.listdir(person_path):
        if image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"  Warning: Could not load {image_name}")
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
            
            if len(faces) == 0:
                print(f"  Warning: No face detected in {image_name} - skipping")
                continue
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_AREA)
                training_data.append(roi_gray)
                labels.append(label_dict[person_name])
                print(f"  Loaded: {image_name} (found {len(faces)} face(s))")

if len(training_data) == 0:
    print("No face images found! Please add images to training_images/ folders.")
    exit()

print(f"\nTraining with {len(training_data)} images...")

# Train the model
training_data = np.array(training_data, dtype=np.uint8)
labels = np.array(labels)

face_recognizer.train(training_data, labels)

# Save model
face_recognizer.write('face_model.yml')

# Save labels
with open('labels.pkl', 'wb') as f:
    pickle.dump(label_dict, f)

print("Model trained successfully!")
print(f"Recognized people: {list(label_dict.keys())}")
print("\nNow run: python test.py")
