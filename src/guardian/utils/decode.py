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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_yolov6(output: np.ndarray, img_size: int, num_classes: int,
                  conf_thresh: float, iou_thresh: float) -> list[Detection]:
    """Decode raw YOLOv6 network output (vectorized).

    Input shape: [num_anchors, 4+1+num_classes] = [cx, cy, w, h, obj, cls...].
    Coordinates are in pixel space.
    Class scores may be probabilities (0-1) or raw logits — auto-detected.
    """
    preds = output.reshape(-1, 4 + 1 + num_classes)

    obj = preds[:, 4]
    cls_raw = preds[:, 5:]
    cls_ids = cls_raw.argmax(axis=1)
    cls_vals = cls_raw[np.arange(len(cls_raw)), cls_ids]

    # Auto-detect: if max class score > 10, it's raw logits needing sigmoid
    if cls_vals.max() > 10.0:
        cls_scores = _sigmoid(cls_vals)
    else:
        cls_scores = cls_vals

    confs = obj * cls_scores

    mask = confs >= conf_thresh
    if not mask.any():
        return []

    preds = preds[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]

    cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    x1 = (cx - w / 2).astype(int)
    y1 = (cy - h / 2).astype(int)
    x2 = (cx + w / 2).astype(int)
    y2 = (cy + h / 2).astype(int)
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    keep = nms(boxes, confs, iou_thresh)
    return [
        Detection(int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]),
                  float(confs[i]), int(cls_ids[i]))
        for i in keep
    ]


def decode_yolov8(output: np.ndarray, scale: float = 1.0,
                  conf_thresh: float = 0.3, iou_thresh: float = 0.5) -> list[Detection]:
    """Decode raw YOLOv8 output (ONNX or OAK blob).

    Handles both shapes:
        [1, 5+, N]  — ONNX format, needs transpose
        [1, N, 5+]  — some OAK blob conversions, already row-major

    scale: preprocessing ratio (IMG_SIZE / max(h, w)). Use 1.0 for OAK
           where camera outputs directly at model input size.
    """
    t = output[0]  # drop batch dim -> (A, B)
    # Detect orientation: features dim is small (5-10), anchors dim is large (1000+)
    # (5, 8400) -> transpose; (8400, 5) -> keep as-is
    if t.shape[1] > t.shape[0]:
        preds = t.T  # (5, 8400) -> (8400, 5)
    else:
        preds = t    # already (8400, 5) or similar

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
