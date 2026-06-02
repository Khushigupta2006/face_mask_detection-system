import cv2
import numpy as np
import time
import pyttsx3
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
model = load_model("models/mask_detector_model.h5")

class_names = ["Mask", "No Mask", "Incorrect"]

# ---------------- VOICE ----------------
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- START CAMERA ----------------
cap = cv2.VideoCapture(0)

last_alert_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (224, 224))
        face_norm = face_resized / 255.0
        face_input = np.expand_dims(face_norm, axis=0)

        pred = model.predict(face_input, verbose=0)
        label_index = np.argmax(pred)
        confidence = np.max(pred) * 100

        # ---------------- LABEL + COLOR ----------------
        if label_index == 0:
            label = "MASK 😷"
            color = (0, 255, 0)

        elif label_index == 1:
            label = "NO MASK ❌"
            color = (0, 0, 255)

        else:
            label = "INCORRECT 😕"
            color = (0, 165, 255)

        # ---------------- ALERT ----------------
        if time.time() - last_alert_time > 3:
            if label_index == 1:
                speak("Please wear a mask")
                last_alert_time = time.time()

            elif label_index == 2:
                speak("Please wear your mask properly")
                last_alert_time = time.time()

        # ---------------- DRAW ----------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            f"{label} ({confidence:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Mask Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()