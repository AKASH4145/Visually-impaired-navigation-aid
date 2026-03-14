import cv2
import numpy as np
import pyttsx3
import time
import threading
from tensorflow.lite.python.interpreter import Interpreter as tflite
import queue
import asyncio
import edge_tts
import pygame
import queue
import os
import tempfile

# TTS Setup 

pygame.mixer.init()
tts_queue = queue.Queue()
last_spoken = {}
COOLDOWN = 3.0

def tts_worker():
    """Single background thread — generates and plays speech one at a time."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            # Save speech to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tmp.close()

            # Generate speech using edge-tts
            asyncio.run(edge_tts.Communicate(text, voice="en-US-AriaNeural").save(tmp.name))

            # Play it
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)

            os.unlink(tmp.name)  # cleanup temp file
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        tts_queue.task_done()

# Start single persistent TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    tts_queue.put(text)

def should_speak(label):
    now = time.time()
    if label not in last_spoken or (now - last_spoken[label]) > COOLDOWN:
        last_spoken[label] = now
        return True
    return False

# Load TFLite model and labels

MODEL_PATH = r'C:\Users\akash\Desktop\OpenCV Projects\Visually impaired navigation aid\Models\detect.tflite'
LABEL_PATH = r'C:\Users\akash\Desktop\OpenCV Projects\Visually impaired navigation aid\Models\labelmap.txt'

interpreter = tflite(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_h, input_w, _ = input_details[0]['shape']

with open(LABEL_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
if CLASSES[0] == '???':          # some labelmap.txt files have this as first line
    CLASSES.pop(0)


# Spatial position 

def get_position(cx, frame_w):
    third = frame_w // 3
    if cx < third:
        return "on your left"
    elif cx < 2 * third:
        return "ahead"
    else:
        return "on your right"
    

# Detection loop


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img, axis=0)

    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes   = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores  = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] < 0.5:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        cx = (x1 + x2) // 2

        label    = CLASSES[int(classes[i])]
        position = get_position(cx, w)
        message  = f"{label} {position}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(frame, f"{message} ({scores[i]:.0%})",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

        # TTS alert

        if should_speak(label):
            speak(message)
            print(f"[ALERT] {message}")

    # fps display
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Nav Aid - Detection Test", frame)

    if cv2.waitKey(1) ==13:
        break

cap.release()
cv2.destroyAllWindows()