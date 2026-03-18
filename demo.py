import cv2
import numpy as np
import time
import threading
from tensorflow.lite.python.interpreter import Interpreter as tflite
import queue
import asyncio
import edge_tts
import pygame
import os
import tempfile

# ── Distance Estimation ───────────────────────────────────────────────────────
def estimate_distance(y1, y2, frame_h):
    ratio = (y2 - y1) / frame_h
    if ratio > 0.6:
        return "very close", (0, 0, 255)      # Red
    elif ratio > 0.35:
        return "close", (0, 165, 255)          # Orange
    elif ratio > 0.15:
        return "nearby", (0, 255, 0)           # Green
    else:
        return "far", (255, 0, 0)              # Blue

# ── TTS Setup ─────────────────────────────────────────────────────────────────
pygame.mixer.init()
tts_queue = queue.Queue()
last_spoken = {}
COOLDOWN = 3.0

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tmp.close()
            asyncio.run(edge_tts.Communicate(text, voice="en-US-AriaNeural").save(tmp.name))
            pygame.mixer.music.load(tmp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            os.unlink(tmp.name)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        tts_queue.task_done()

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

# ── Load TFLite Model ─────────────────────────────────────────────────────────
MODEL_PATH = r'C:\Users\akash\Desktop\OpenCV Projects\Visually impaired navigation aid\Models\detect.tflite'
LABEL_PATH = r'C:\Users\akash\Desktop\OpenCV Projects\Visually impaired navigation aid\Models\labelmap.txt'

interpreter = tflite(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_h, input_w, _ = input_details[0]['shape']

with open(LABEL_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]
if CLASSES[0] == '???':
    CLASSES.pop(0)

# ── Spatial Position ──────────────────────────────────────────────────────────
def get_position(cx, frame_w):
    third = frame_w // 3
    if cx < third:
        return "on your left"
    elif cx < 2 * third:
        return "ahead"
    else:
        return "on your right"

# ── On Screen HUD ─────────────────────────────────────────────────────────────
def draw_hud(frame, detections_list, fps):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 40 + len(detections_list) * 22 + 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, "Detected:", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    for idx, (label, position, distance, color) in enumerate(detections_list):
        cv2.putText(frame, f"  {label} | {position} | {distance}",
                    (10, 66 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Zone lines at bottom
    cv2.line(frame, (w // 3, h - 40), (w // 3, h), (255, 255, 255), 1)
    cv2.line(frame, (2 * w // 3, h - 40), (2 * w // 3, h), (255, 255, 255), 1)
    cv2.putText(frame, "LEFT",  (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "AHEAD", (w // 3 + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "RIGHT", (2 * w // 3 + 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ── Detection Loop ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)  # 1 = DroidCam, 0 = built-in webcam
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detections_list = []  #  reset every frame

    # Preprocess
    img = cv2.resize(frame, (input_w, input_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img, axis=0)
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Inference
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
        distance, color = estimate_distance(y1, y2, h)

        # Proximity warning
        if distance == "very close":
            message = f"Warning! {label} very close {position}"
        else:
            message = f"{label} {distance} {position}"

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} | {distance} ({scores[i]:.0%})",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # TTS
        if should_speak(label):
            speak(message)
            print(f"[ALERT] {message}")

        detections_list.append((label, position, distance, color))  # ✅ always append

    # FPS — calculated manually for accuracy
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    #  Draw HUD before imshow
    draw_hud(frame, detections_list, fps)

    #  Single imshow and waitKey
    cv2.imshow("Visually Impaired Navigation Aid", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Graceful Exit ─────────────────────────────────────────────────────────────
tts_queue.put(None)       # stop TTS worker thread
cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("[INFO] Exited cleanly.")