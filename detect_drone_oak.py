"""
Drone detection on OAK-1W using trained YOLOv6n blob

Requirements:
    pip install depthai

Usage:
    python detect_drone_oak.py
    python detect_drone_oak.py --blob path/to/model.blob
"""

import argparse
from pathlib import Path

import depthai as dai
import cv2
import numpy as np


DEFAULT_BLOB = Path("drone_yolov6n.blob")
IMG_SIZE = 416
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
LABELS = ["drone"]


def build_pipeline(blob_path: Path) -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(IMG_SIZE, IMG_SIZE)
    cam.setInterleaved(False)
    cam.setFp16(True)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(blob_path))
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    cam.preview.link(nn.input)
    cam.preview.link(xout_rgb.input)
    nn.out.link(xout_nn.input)

    return pipeline


def decode_yolov6(output: np.ndarray, conf_thresh: float, iou_thresh: float):
    """Decode raw YOLOv6 output into bounding boxes."""
    detections = []
    # YOLOv6n output shape: [1, num_anchors, 5+nc]
    output = output.reshape(-1, 5 + len(LABELS))

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
        x1 = int((cx - w / 2) * IMG_SIZE)
        y1 = int((cy - h / 2) * IMG_SIZE)
        x2 = int((cx + w / 2) * IMG_SIZE)
        y2 = int((cy + h / 2) * IMG_SIZE)
        detections.append((x1, y1, x2, y2, confidence, cls_id))

    return detections


def run(blob_path: Path) -> None:
    pipeline = build_pipeline(blob_path)

    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

        print("OAK-1W running — press Q to quit")

        while True:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if in_nn is not None and in_rgb is not None:
                output = np.array(in_nn.getFirstLayerFp16())
                detections = decode_yolov6(output, CONF_THRESHOLD, IOU_THRESHOLD)

                for x1, y1, x2, y2, conf, cls_id in detections:
                    label = f"{LABELS[cls_id]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    print(f"Detected: {label} | Box: [{x1}, {y1}, {x2}, {y2}]")

            if in_rgb is not None:
                cv2.imshow("OAK-1W Drone Detection", frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv6n Drone Detection on OAK-1W")
    parser.add_argument("--blob", type=Path, default=DEFAULT_BLOB, help="Path to .blob model file")
    args = parser.parse_args()

    if not args.blob.exists():
        print(f"ERROR: Blob not found at {args.blob}")
        print("Run convert_to_blob.py first.")
        return

    run(args.blob)


if __name__ == "__main__":
    main()
