"""YOLO output decoding and NMS for YOLOv6 and YOLOv8 formats."""

from typing import NamedTuple

import numpy as np


class Detection(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Non-Maximum Suppression. Returns indices of boxes to keep."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thresh]
    return keep


def decode_yolov6(output: np.ndarray, img_size: int, num_classes: int,
                  conf_thresh: float, iou_thresh: float) -> list[Detection]:
    """Decode raw YOLOv6 network output.

    Input shape: [1, num_anchors, 5+num_classes] or flat equivalent.
    Coordinates are normalized (0-1) relative to img_size.
    """
    output = output.reshape(-1, 5 + num_classes)
    detections = []

    for det in output:
        obj_conf = float(det[4])
        if obj_conf < conf_thresh:
            continue
        class_scores = det[5:]
        cls_id = int(np.argmax(class_scores))
        confidence = obj_conf * float(class_scores[cls_id])
        if confidence < conf_thresh:
            continue
        cx, cy, w, h = det[:4]
        x1 = int((cx - w / 2) * img_size)
        y1 = int((cy - h / 2) * img_size)
        x2 = int((cx + w / 2) * img_size)
        y2 = int((cy + h / 2) * img_size)
        detections.append(Detection(x1, y1, x2, y2, confidence, cls_id))

    # Apply NMS
    if len(detections) <= 1:
        return detections
    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=np.float32)
    scores = np.array([d.confidence for d in detections], dtype=np.float32)
    keep = nms(boxes, scores, iou_thresh)
    return [detections[i] for i in keep]


def decode_yolov8(output: np.ndarray, scale: float,
                  conf_thresh: float, iou_thresh: float) -> list[Detection]:
    """Decode raw YOLOv8 ONNX output.

    Input shape: [1, 5, 8400] for single-class = [cx, cy, w, h, score] transposed.
    scale: the ratio used during preprocessing (IMG_SIZE / max(h, w)).
    """
    preds = output[0].T  # [8400, 5+]

    if preds.shape[1] == 5:
        # Single class: [cx, cy, w, h, score]
        scores = preds[:, 4]
        class_ids = np.zeros(len(preds), dtype=int)
    else:
        # Multi-class: [cx, cy, w, h, cls0, cls1, ...]
        class_ids = np.argmax(preds[:, 4:], axis=1)
        scores = preds[np.arange(len(preds)), 4 + class_ids]

    mask = scores >= conf_thresh
    preds, scores, class_ids = preds[mask], scores[mask], class_ids[mask]
    if len(preds) == 0:
        return []

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = ((cx - w / 2) / scale).astype(int)
    y1 = ((cy - h / 2) / scale).astype(int)
    x2 = ((cx + w / 2) / scale).astype(int)
    y2 = ((cy + h / 2) / scale).astype(int)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    keep = nms(boxes, scores, iou_thresh)
    return [
        Detection(int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]),
                  float(scores[i]), int(class_ids[i]))
        for i in keep
    ]


def preprocess_frame(frame: np.ndarray, img_size: int) -> tuple[np.ndarray, float]:
    """Preprocess a frame for YOLO inference.

    Returns (blob in NCHW format, scale factor).
    """
    h, w = frame.shape[:2]
    scale = img_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    import cv2
    resized = cv2.resize(frame, (nw, nh))

    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    return blob, scale
