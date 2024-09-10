import cv2
import numpy as np
import sqlite3

# Initialize face recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read(r'D:\A\Facial Recognition System Opencv Based On Raspberry Pi 3 in Realtime\trainer\trainer.yml')
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Load the face detector
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Initialize the camera
cam = cv2.VideoCapture(0)

# Connect to SQLite database
db = sqlite3.connect("test.db")
curs = db.cursor()

# Create table if not exists
curs.execute('''
    CREATE TABLE IF NOT EXISTS facebase (
        npm TEXT PRIMARY KEY,
        nama TEXT
    )
''')
db.commit()

def getProfile(id):
    curs = db.cursor()
    cmd = "SELECT * FROM facebase WHERE npm=?"
    curs.execute(cmd, (id,))
    profile = curs.fetchone()
    curs.close()
    return profile

while True:
    ret, im = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (225, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        profile = getProfile(id)
        print(f"ID: {id}, Confidence: {conf}")

        if conf < 43:
            if profile is not None:
                cv2.putText(im, str(profile[0]), (x+90, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(im, str(profile[1]), (x+90, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                id = "Unknown"
                cv2.putText(im, str(id), (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

    cv2.imshow('Face Recognition', im)

    if cv2.waitKey(10) & 0xFF == ord('q'): #q --> for Quit
        break

cam.release()
cv2.destroyAllWindows()
