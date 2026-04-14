## Visually Impaired Navigation Aid 

 A real-time assistive navigation system designed to help visually impaired individuals navigate everyday environments safely. The system captures live video from a webcam, detects objects using a TFLite MobileNet-SSD model, and determines their spatial position (left, ahead, or right) relative to the user.

 Detected objects are announced through natural-sounding voice alerts using Microsoft's Neural Text to speech (TTS) engine. The system includes a cooldown mechanism to prevent repetitive alerts for the same object class.
 
  Designed to be lightweight and efficient, the project runs entirely on a standard laptop CPU without requiring any GPU, making it accessible and practical for real-world use.

---

## Motivation

 Visually impaired individuals face significant challenges when navigating unfamiliar or dynamic environments independently. Existing assistive technologies are often expensive, bulky, or require specialized hardware. This project explores how modern computer vision and text-to-speech technologies can be combined into a simple, affordable, and accessible navigation aid. By leveraging object detection and spatial audio feedback, the system aims to provide real-time environmental awareness through sound instead of sight. 
 
 The goal is to demonstrate how lightweight AI models running on everyday hardware can meaningfully improve the quality of life for visually impaired individuals without relying on complex or costly infrastructure.

---

##  Features

 - Real-time object detection

 - Spatial position announcement (left, ahead, right)

 - Natural voice alerts using Microsoft Neural TTS

 - Cooldown system per object class

 - Lightweight — runs on CPU without GPU

 - Works with webcam feed


---

##  Concepts Used

 - Object Detection using TFLite MobileNet-SSD

 - Text-to-Speech audio alerts

 - Spatial positioning (left / ahead / right)

 - Cooldown system to avoid repeated alerts

 - Real-time video processing


---

##  Tech Stack

 - Language — Python

 - Computer Vision — OpenCV

 - Object Detection — TFLite MobileNet-SSD

 - TTS — edge-tts (Microsoft Neural Voice)

 - Audio Playback — pygame

 - Numerical Processing — NumPy 

---

##  System Architecture

Workflow:
1. Live video feed (webcam)
2. Frame Input
3. Preprocesing
4. Object Detection (TFLite MobileNet-SSD — detect.tflite )
5. Confidence Filtering ( > 50% )
6. Spatial Classification (Left , Right , Ahead)
7. Cooldown Check
8. TTS alert ( edge tts generates audio , pygame plays it)
9. Visual Overlay (bounding boxes and label)
10. Display our live window

---

##  Project Structure

```text

Visually Impaired Navigation Aid/
├── main.py
├── requirements.txt
├── Models/
│   ├── detect.tflite
│   └── labelmap.txt
└── README.md

```
---

## Setup & Run

- git clone https://github.com/AKASH4145/Visually-impaired-navigation-aid
- cd   Visually-impaired-navigation-aid
- pip install -r requirements.txt  
- python main.py

---

## Demo Screenshots and Video
 
 🎥 Demo Video :[Watch on drive](https://drive.google.com/file/d/1riByIPXL4uLFiqIOJKDWfvIDHHPTYyvM/view?usp=sharing)

---

## Observations

- MobileNet-SSD provides fast, lightweight detection suitable for real-time use

- Spatial positioning helps users understand object location without vision

- Cooldown system prevents overwhelming the user with repeated alerts

- edge-tts provides natural sounding voice compared to pyttsx3

---

## Limitations

- Detection accuracy depends on lighting conditions

- Depth/distance estimation not yet implemented

- Works best with a stable front-facing camera

- Requires internet connection for edge-tts voice generation

---

## Future Scope


- Obstacle proximity warning

- Mobile phone camera support

- Offline TTS support

- Raspberry Pi deployment

---

## Applications 

- Assistive Navigation — Helps visually impaired individuals detect and avoid obstacles in real-time

- Healthcare & Rehabilitation — Can be used in hospitals or rehab centers to assist patients with visual impairments

- Elderly Assistance — Supports elderly individuals with deteriorating vision to navigate safely

- Educational Tool — Demonstrates how AI and computer vision can solve real-world accessibility problems

- Robotics — Object detection and spatial awareness pipeline can be adapted for autonomous robots

- Warehouse Automation — Detect and announce nearby objects or obstacles in industrial environments

- ADAS Research — Spatial object detection concepts are directly applicable to driver assistance systems

- Computer Vision Research — Serves as a base project for exploring lightweight edge AI deployment

--- 

## Author

Akash GS | Mechanical Engineering student exploring AI, computer vision, and applied Python development

---