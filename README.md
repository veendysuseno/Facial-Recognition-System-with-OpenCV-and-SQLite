# Facial Recognition System with OpenCV and SQLite

This project implements a facial recognition system using OpenCV and SQLite. It includes three main components: face registration, model training, and face recognition. The system captures face images, stores user information in a SQLite database, trains a face recognizer, and then uses the trained model to recognize faces in real-time.

## Project Components

1. **register.py**:

   - Captures face images from the webcam and saves them to the `dataSet` directory.
   - Stores user information in an SQLite database (`test.db`).
   - Ensures that each user has multiple face images for better training.

2. **trainner.py**:

   - Reads face images from the `dataSet` directory.
   - Trains the LBPH (Local Binary Patterns Histogram) face recognizer.
   - Saves the trained model to `trainer/trainer.yml`.

3. **facerec.py**:
   - Loads the trained face recognizer model.
   - Performs real-time face recognition using the webcam.
   - Displays the recognized user's ID and name on the video feed.

## Setup Instructions

1. **Install Dependencies**:
   Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

2. Download Haar Cascade: Make sure you have the Haar Cascade XML file (haarcascade_frontalface_default.xml). You can download it from the OpenCV GitHub repository.

3. Run Registration: Execute register.py to capture face images and store user information:
   ```bash
   python register.py
   ```
4. Train the Model: Run trainner.py to train the face recognizer model:
   ```bash
   python trainner.py
   ```
5. Run Face Recognition: Start facerec.py to perform real-time face recognition:
   ```bah
   python facerec.py
   ```

## Troubleshooting

- No Faces Detected: Ensure the face images are clear and properly lit. Adjust the detectMultiScale parameters in trainner.py and facerec.py if necessary.
- Model Not Saving: Verify that the trainer directory exists and has write permissions.

<br/>

@Copyright 2020 | Veendy <br/>
