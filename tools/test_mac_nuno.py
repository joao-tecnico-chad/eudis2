"""Run Nuno's YOLOv8n detection on Mac using webcam + ONNX."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from guardian.utils.decode import decode_yolov8, preprocess_frame

ONNX_PATH = "models/nuno_yolov8n.onnx"
IMG_SIZE = 640
CONF_THRESH = 0.5
IOU_THRESH = 0.5
LABELS = ["drone"]

session = ort.InferenceSession(ONNX_PATH, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"Loaded {ONNX_PATH} — providers: {session.get_providers()}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    sys.exit(1)

print("Running Nuno's YOLOv8n on Mac webcam — press 'q' to quit")

fps_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob, scale = preprocess_frame(frame, IMG_SIZE)
    output = session.run(None, {input_name: blob})[0]  # [1, 5, 8400]
    detections = decode_yolov8(output, scale, CONF_THRESH, IOU_THRESH)

    for det in detections:
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
        label = f"{LABELS[det.class_id]} {det.confidence:.2f}"
        cv2.putText(frame, label, (det.x1, det.y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_count += 1
    elapsed = time.time() - fps_time
    if elapsed >= 2.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Nuno YOLOv8n Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
