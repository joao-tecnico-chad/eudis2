"""Run drone detection on Mac using webcam + ONNX model.

Supports both YOLOv6 and YOLOv8 ONNX models with live webcam display.
Uses the same decode functions as the OAK pipeline for consistency.

Usage:
    python tools/test_mac_detection.py
    python tools/test_mac_detection.py --model-format yolov8 --onnx models/nuno_yolov8n.onnx
    python tools/test_mac_detection.py --conf 0.3 --source video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from guardian.utils.decode import decode_yolov6, decode_yolov8, preprocess_frame

DEFAULT_ONNX = {
    "yolov6": "models/drone_yolov6n.onnx",
    "yolov8": "models/nuno_yolov8n.onnx",
}

parser = argparse.ArgumentParser(description="Mac webcam drone detection")
parser.add_argument("--model-format", choices=["yolov6", "yolov8"], default="yolov6")
parser.add_argument("--onnx", default=None, help="Path to ONNX model")
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
parser.add_argument("--source", default="0", help="Webcam index or video file")
args = parser.parse_args()

onnx_path = args.onnx or DEFAULT_ONNX[args.model_format]
providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession(onnx_path, providers=providers)
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
img_size = input_shape[2] if len(input_shape) >= 3 else 416

print(f"Model:  {args.model_format} ({onnx_path})")
print(f"Input:  {img_size}x{img_size}  conf>{args.conf}")
print(f"Provider: {session.get_providers()[0]}")

source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"Cannot open source: {args.source}")
    sys.exit(1)

print("Press 'q' to quit")

fps_time = time.time()
frame_count = 0
fps_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob, scale = preprocess_frame(frame, img_size)
    output = session.run(None, {input_name: blob})[0]

    if args.model_format == "yolov8":
        detections = decode_yolov8(output, scale, args.conf, args.iou)
    else:
        detections = decode_yolov6(output, img_size, 1, args.conf, args.iou)

    for det in detections:
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        label = f"drone {det.confidence:.2f}"
        cv2.putText(frame, label, (det.x1, det.y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_count += 1
    elapsed = time.time() - fps_time
    if elapsed >= 2.0:
        fps_text = f"FPS: {frame_count / elapsed:.1f}"
        frame_count = 0
        fps_time = time.time()
    if fps_text:
        cv2.putText(frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drone Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
