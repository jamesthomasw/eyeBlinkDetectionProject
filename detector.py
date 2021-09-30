import tensorflow as tf
import numpy as np
import os
import cv2
import dlib
import time
import imutils
import matplotlib.pyplot as plt

def split_blink_data(blink_data):
    durasi = []
    frekuensi = []

    for duration, frequency in blink_data:
        durasi.append(duration)
        frekuensi.append(frequency)

    return durasi, frekuensi

def calculate_fps(frame, start, end):
    time_diff = end - start
    if time_diff == 0:
        framepersecond = 0.0
    else:
        framepersecond = 1/time_diff
        
    fps_text = "FPS: {:.2f}".format(framepersecond)
    cv2.putText(frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

def put_total_frames(frame, total_frames):
    frame_text = "Total Frames: {:.0f}".format(total_frames)
    cv2.putText(frame, frame_text, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, total_frames


def calculate_perclos(frame, perclos_frames, total_frames):
    perclos = perclos_frames / total_frames
    perclos_text = "PERCLOS: {:.2f}".format(perclos)
    cv2.putText(frame, perclos_text, (480, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

drowsyModel = tf.keras.models.load_model("saved_model/mobilenet_128_v2.h5", compile=False)

eyes_status = ""

blink_frames = 0
blink_total = 0
total_frames = 0
perclos_frames = 0
microsleep = 0

blink_data = []

cap = cv2.VideoCapture(0)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = cap.read()
    if frame is None:
        break
        
    start = time.time()
    
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        fx1, fy1, fx2, fy2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        
        landmarks = predictor(gray, face)
        x1 = landmarks.part(37).x
        y1 = landmarks.part(37).y
        x2 = landmarks.part(46).x
        y2 = landmarks.part(46).y

        cv2.rectangle(frame, (x1-18, y1-20), (x2+18, y2+20), (0, 255, 0), 2)
        
        eyes_roi = frame[y1-20:y2+20, x1-18:x2+18]

        roi_resized = cv2.resize(eyes_roi, (128, 128))
        roi_final = np.expand_dims(roi_resized, axis=0)
        roi_final = roi_final/255.0

        prediction = drowsyModel.predict(roi_final)
        if round(float(prediction)) == 1:
            eyes_status = "Open"
        else:
            eyes_status = "Close"

        cv2.putText(frame, f'Both Eyes State: {eyes_status}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    if (eyes_status == "Close"):
        blink_frames += 1
        perclos_frames += 1
        cv2.putText(frame, "BLINKING", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        if (blink_frames >= 2):
            blink_total += 1
            blink_duration = round(float(blink_frames * (1/frame_rate)), 3)
            if (blink_duration > 0.5):
                microsleep += 1
            blink_data.append([blink_duration, blink_total])
        blink_frames = 0
            
    cv2.putText(frame, f'Blink: {blink_total}', (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Microsleep: {microsleep}', (480, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    (durasi, frekuensi) = split_blink_data(blink_data)

    total_frames += 1
    (frame, total_frames) = put_total_frames(frame, total_frames)
    frame = calculate_perclos(frame, perclos_frames, total_frames)
        
    end = time.time()
    frame = calculate_fps(frame, start, end)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
