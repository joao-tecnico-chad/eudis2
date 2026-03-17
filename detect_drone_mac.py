"""
Drone detection on MacBook using trained YOLOv6n (ONNX)
Uses MPS (Apple GPU) via onnxruntime-silicon when available

Usage:
    python detect_drone_mac.py                   # webcam
    python detect_drone_mac.py --source video.mp4
    python detect_drone_mac.py --source image.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = Path("best.onnx")
IMG_SIZE = 416
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5


def build_session() -> ort.InferenceSession:
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    # Prefer CoreML (Apple GPU) > CPU
    preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    available = [p for p in preferred if p in providers]
    print(f"Using: {available[0]}")
    return ort.InferenceSession(str(MODEL_PATH), providers=available)


def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    scale = IMG_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))

    # Pad to square
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    return blob, scale, nw, nh


def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(output: np.ndarray, scale: float, conf_thresh: float, iou_thresh: float):
    # YOLOv6 ONNX output: [1, num_anchors, 6] = [cx, cy, w, h, obj_conf, cls_conf]
    preds = output[0]  # [num_anchors, 6]

    scores = preds[:, 4] * preds[:, 5]
    mask = scores >= conf_thresh
    preds, scores = preds[mask], scores[mask]
    if len(preds) == 0:
        return []

    # Convert cx, cy, w, h -> x1, y1, x2, y2 and rescale
    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - w / 2) / scale
    y1 = (cy - h / 2) / scale
    x2 = (cx + w / 2) / scale
    y2 = (cy + h / 2) / scale
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, scores, iou_thresh)
    return [(boxes[i].astype(int).tolist(), float(scores[i]), 0) for i in keep]


def run(source: str) -> None:
    session = build_session()
    input_name = session.get_inputs()[0].name

    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        frame = cv2.imread(source)
        blob, scale, _, _ = preprocess(frame)
        output = session.run(None, {input_name: blob})[0]
        detections = postprocess(output, scale, CONF_THRESHOLD, IOU_THRESHOLD)
        for (x1, y1, x2, y2), conf, _ in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"drone {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Drone Detection", frame)
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob, scale, _, _ = preprocess(frame)
        output = session.run(None, {input_name: blob})[0]
        detections = postprocess(output, scale, CONF_THRESHOLD, IOU_THRESHOLD)

        for (x1, y1, x2, y2), conf, _ in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"drone {conf:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"drone {conf:.2f} [{x1},{y1},{x2},{y2}]")

        now = time.time()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Drone Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index, video file, or image path")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    run(args.source)


if __name__ == "__main__":
    main()
