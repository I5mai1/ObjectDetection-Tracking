# Real-Time Object Detection using YOLO and OpenCV

This project demonstrates real-time object detection using the YOLO (You Only Look Once) model and OpenCV.

## Features

- Real-time object detection using the YOLO model.
- Displays bounding boxes around detected objects.
- Labels objects with class names and confidence scores.
- Removes redundant detections using non-maximum suppression.

## Prerequisites

- Python 3.x
- OpenCV
- NumPy
- YOLO configuration and weights files (e.g., yolov3.cfg and yolov3.weights)
- COCO class names file (coco.names)

## Installation

1. Install Python 3.x
2. Install required dependencies:
   ```bash
   pip install opencv-python numpy
   
Usage

    Download YOLO configuration file (yolov3.cfg), weights file (yolov3.weights), and class names file (coco.names) from YOLO website.
    Organize files in the project directory.
    Run the script:

    bash

python yolo_detection.py

Press 'q' to exit the application.
