from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import pandas as pd
from datetime import date, datetime
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask(__name__)

# Number of images to capture per user
nimgs = 10

# Date formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Get absolute path for Haar Cascade file
haar_cascade_path = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')

# Load Haar Cascade for face detection
face_detector = cv2.CascadeClassifier(haar_cascade_path)

if face_detector.empty():
    raise IOError(f"Haar Cascade file not found at {haar_cascade_path}. Ensure the file exists and is accessible.")

# Ensure necessary directories exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Initialize today's attendance file
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time\n')

# Function to get total registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract attendance
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], len(df)

# Function to add attendance
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in df['Roll'].values:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

# Function to extract faces
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

# Function to identify a face
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Function to train the model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Home route
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Add user route
@app.route('/add', methods=['POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    user_folder = f'static/faces/{newusername}_{newuserid}'
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while count < nimgs:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (50, 50))
            cv2.imwrite(f'{user_folder}/{count}.jpg', face_resized)
            count += 1
            if count >= nimgs:
                break
    cap.release()
    train_model()
    return render_template('home.html', mess="User added successfully!", totalreg=totalreg(), datetoday2=datetoday2)

# Start attendance route
@app.route('/start')
def start():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cap.release()
            return render_template('home.html', mess=f"Attendance marked for {identified_person}!", totalreg=totalreg(), datetoday2=datetoday2)
        else:
            cap.release()
            return render_template('home.html', mess="No face detected! Please try again.", totalreg=totalreg(), datetoday2=datetoday2)
    cap.release()
    return render_template('home.html', mess="No face detected!", totalreg=totalreg(), datetoday2=datetoday2)

if __name__ == '__main__':
    app.run(debug=True)
