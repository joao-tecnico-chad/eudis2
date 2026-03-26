"""Run YOLOv6n detection on Mac using webcam + PyTorch weights."""

import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "YOLOv6"))

from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression

WEIGHTS = "models/best_yolov6n.pt"
IMG_SIZE = 640
CONF_THRESH = 0.5
IOU_THRESH = 0.5
LABELS = ["drone"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = DetectBackend(WEIGHTS, device=DEVICE)
model.model.float().eval()
print(f"Loaded {WEIGHTS} on {DEVICE}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    sys.exit(1)

print("Running YOLOv6n on Mac webcam — press 'q' to quit")

fps_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    h, w = frame.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    tensor = torch.from_numpy(blob).to(DEVICE)

    # Inference
    with torch.no_grad():
        pred = model(tensor)

    # NMS
    det = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, max_det=10)[0]

    if len(det):
        for *xyxy, conf, cls in det:
            x1 = int(xyxy[0] / scale)
            y1 = int(xyxy[1] / scale)
            x2 = int(xyxy[2] / scale)
            y2 = int(xyxy[3] / scale)
            label = f"{LABELS[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_count += 1
    elapsed = time.time() - fps_time
    if elapsed >= 2.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLOv6n Detection", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
