# Raspberry Pi Waste Detection

## Final Year Capstone Project

A real-time computer vision system that detects and classifies potential waste objects using YOLOv5 Nano and OpenCV, deployed for lightweight edge AI inference on a Raspberry Pi.

This project explores how far a small, resource-constrained device can go running modern object detection models in real time — no cloud, no GPU, just a Pi and a camera.

## Key Features:
Real-time object detection on live camera feed
YOLOv5 Nano — a lightweight model optimized for edge devices
OpenCV for video capture, drawing, and display
Bounding-box and confidence-score visualization
Simple waste / non-waste categorization logic
Runs entirely on-device — no internet connection required for inference

## Hardware Requirements:
Raspberry Pi:	Pi 4 (4GB+) recommended; Pi 3B+ will run but slower

Camera:	USB webcam or Pi Camera Module

Storage:	16GB+ microSD (32GB recommended)

OS:	Raspberry Pi OS (64-bit recommended)

## Installation:
#Clone the repository
```bash
git clone https://github.com/anshr99/raspberry-pi-waste-detection.git
cd raspberry-pi-waste-detection

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage:

Connect a USB webcam (or enable the Pi Camera Module), then run:

```bash
python waste_detection.py
```

A live window will open showing the camera feed with bounding boxes and waste/not-waste labels.

Press q to quit.

## How It Works:
Camera Feed → YOLOv5 Nano Inference → Bounding Boxes + Class Labels → Waste/Non-Waste Mapping → Display

Frames are captured from the camera using OpenCV.

Each frame is passed through a YOLOv5 Nano model (yolov5n.pt) for object detection.

Detected classes are checked against a waste-related label list (e.g. bottle, cup) to flag them as "Waste."

Bounding boxes, labels, and confidence scores are drawn on the frame and displayed in real time.

