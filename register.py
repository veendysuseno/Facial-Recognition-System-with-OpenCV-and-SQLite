import sqlite3
import cv2

# Initialize camera
cam = cv2.VideoCapture(0)

# Load face detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

def insertOrUpdate(id, nama):
    curs.execute("SELECT * FROM facebase WHERE npm=?", (id,))
    statusData = curs.fetchone()
    if statusData:
        curs.execute("UPDATE facebase SET nama=? WHERE npm=?", (nama, id))
    else:
        curs.execute("INSERT INTO facebase (npm, nama) VALUES (?, ?)", (id, nama))
    db.commit()

# Input ID and Name
id = input('Enter your ID: ')
nama = input('Enter your Name: ')
insertOrUpdate(id, nama)

# Capture faces and save images
sampleNum = 0
while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    # Process detected faces
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"dataSet/User.{id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
