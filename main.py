import cv2
import numpy as np
import pyttsx3
import time
import threading

# Speak Setup

engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

tts_lock = threading.Lock()
last_spoken = {}          # tracks last time each class was announced
COOLDOWN = 3.0            # seconds before repeating the same alert

def speak(text):
    """Run TTS in a background thread so video doesn't freeze."""
    def _speak():
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def should_speak(label):
    """Only speak if cooldown has passed for this class."""
    now = time.time()
    if label not in last_spoken or (now - last_spoken[label]) > COOLDOWN:
        last_spoken[label] = now
        return True
    return False

# Load model

net = cv2.dnn.readNetFromTensorflow('Models/detect.tflite')

with open('models/labelmap.txt', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

#Detection loop

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, size=(300, 300),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            label = f"{CLASSES[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Nav Aid - Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()