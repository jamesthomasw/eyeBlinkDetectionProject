import tensorflow as tf
import numpy as np
import cv2
import imutils
import time
import os


def split_blink_data(blink_data):
    durasi = []
    frekuensi = []

    for duration, frequency in blink_data:
        durasi.append(duration)
        frekuensi.append(frequency)

    return durasi, frekuensi


def calc_fps(frame, start, end):
    time_diff = end - start
    if time_diff == 0:
        framepersecond = 0.0
    else:
        framepersecond = 1 / time_diff

    fps_text = "FPS: {:.2f}".format(framepersecond)
    cv2.putText(frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def put_total_frames(frame, total_frames):
    frame_text = "Total Frames: {:.0f}".format(total_frames)
    cv2.putText(frame, frame_text, (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, total_frames


def calc_perclos(frame, perclos_frames, total_frames):
    perclos = perclos_frames / total_frames
    perclos_text = "PERCLOS: {:.2f}".format(perclos)
    cv2.putText(frame, perclos_text, (640, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def nothing(x):
    pass


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

drowsyModel = tf.keras.models.load_model("mobnet_drowsy_v8-9.h5", compile=False)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
leftEye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
rightEye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

leftEyes_status = ''
rightEyes_status = ''

blink_frames = 0
blink_total = 0
total_frames = 0
perclos_frames = 0
microsleep = 0

blink_data = []

cv2.namedWindow("Trackbar", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Trackbar", 400, 200)
cv2.createTrackbar("Face Scale", "Trackbar", 12, 20, nothing)
cv2.createTrackbar("Face Neighbours", "Trackbar", 4, 20, nothing)
cv2.createTrackbar("Eye Scale", "Trackbar", 11, 20, nothing)
cv2.createTrackbar("Eye Neighbours", "Trackbar", 5, 20, nothing)

cap = cv2.VideoCapture("20181018_093050_NF.mp4")
frame_rate = cap.get(cv2.CAP_PROP_FPS)

video_start = time.time()

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    start = time.time()

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_scale = cv2.getTrackbarPos("Face Scale", "Trackbar")
    face_neighbours = cv2.getTrackbarPos("Face Neighbours", "Trackbar")

    faces = face_cascade.detectMultiScale(gray, face_scale / 10, face_neighbours)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eye_scale = cv2.getTrackbarPos("Eye Scale", "Trackbar")
        eye_neighbours = cv2.getTrackbarPos("Eye Neighbours", "Trackbar")

        left_eyes = leftEye_cascade.detectMultiScale(roi_gray, eye_scale / 10, eye_neighbours)
        if len(left_eyes) == 0:
            leftEyes_status = ""
        else:
            for (lex, ley, lew, leh) in left_eyes:
                leftEyes_roi = roi_color[ley:ley + leh, lex:lex + lew]
                cv2.rectangle(roi_color, (lex, ley), (lex + lew, ley + leh), (0, 255, 0), 2)

            leftEyes_resized = cv2.resize(leftEyes_roi, (224, 224))
            leftEyes_final = np.expand_dims(leftEyes_resized, axis=0)
            leftEyes_final = leftEyes_final / 255.0

            leftEyes_output = drowsyModel.predict(leftEyes_final)
            if round(float(leftEyes_output)) == 1:
                leftEyes_status = "Open"
            else:
                leftEyes_status = "Close"

        right_eyes = rightEye_cascade.detectMultiScale(roi_gray, eye_scale / 10, eye_neighbours)
        if len(right_eyes) == 0:
            rightEyes_status = ""
        else:
            for (rex, rey, rew, reh) in right_eyes:
                rightEyes_roi = roi_color[rey:rey + reh, rex:rex + rew]
                cv2.rectangle(roi_color, (rex, rey), (rex + rew, rey + reh), (0, 255, 0), 2)

            rightEyes_resized = cv2.resize(rightEyes_roi, (224, 224))
            rightEyes_final = np.expand_dims(rightEyes_resized, axis=0)
            rightEyes_final = rightEyes_final / 255.0

            rightEyes_output = drowsyModel.predict(rightEyes_final)
            if round(float(rightEyes_output)) == 1:
                rightEyes_status = "Open"
            else:
                rightEyes_status = "Close"

    if (leftEyes_status == "Open" and rightEyes_status == "Open"):
        cv2.putText(frame, "Both eyes are opened", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif (leftEyes_status == "Close" and rightEyes_status == "Close"):
        cv2.putText(frame, "Both eyes are closed", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Eyes are not detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f'Left Eye State: {leftEyes_status}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Right Eye State: {rightEyes_status}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if (leftEyes_status == "Close" and rightEyes_status == "Close"):
        blink_frames += 1
        perclos_frames += 1
        cv2.putText(frame, "BLINKING", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        if (blink_frames >= 2):
            blink_total += 1
            blink_duration = round(float(blink_frames * (1 / frame_rate)), 3)
            if (blink_duration > 0.5):
                microsleep += 1
            blink_data.append([blink_duration, blink_total])
        blink_frames = 0

    cv2.putText(frame, f'Blink: {blink_total}', (20, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f'Microsleep: {microsleep}', (640, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    (durasi, frekuensi) = split_blink_data(blink_data)

    total_frames += 1
    (frame, total_frames) = put_total_frames(frame, total_frames)
    frame = calc_perclos(frame, perclos_frames, total_frames)

    end = time.time()
    frame = calc_fps(frame, start, end)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

video_stop = time.time()
diff = video_stop - video_start
print(diff)
print(durasi)

cap.release()
cv2.destroyAllWindows()