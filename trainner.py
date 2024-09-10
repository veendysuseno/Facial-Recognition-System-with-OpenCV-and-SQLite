import cv2
import os
import numpy as np
from PIL import Image

# Initialize face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def getImagesAndLabels(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The directory {path} does not exist.")

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(imageNp, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print(f"No faces detected in image {imagePath}")
                
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                ids.append(Id)
        except Exception as e:
            print(f"Skipping file {imagePath} due to error: {e}")
    
    return faceSamples, ids

# Use absolute path if necessary
dataSetPath = r'D:\A\Facial Recognition System Opencv Based On Raspberry Pi 3 in Realtime\dataSet'

# Ensure the trainer directory exists
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Check if the directory exists
if not os.path.exists(dataSetPath):
    print(f"Error: The directory {dataSetPath} does not exist.")
else:
    # Get training data
    faces, ids = getImagesAndLabels(dataSetPath)

    if len(faces) > 0 and len(ids) > 0:
        # Train the recognizer and save the model
        recognizer.train(faces, np.array(ids))
        recognizer.save('trainer/trainer.yml')
        print("Training complete and model saved.")
    else:
        print("No data to train the model. Make sure you have sufficient face samples.")
