"""
Drone detection using Ultralytics YOLO with native MPS (Apple GPU)

Features:
    - ByteTrack object tracking (persistent drone IDs)
    - Temporal filtering (only show drones seen N+ consecutive frames)
    - Distance estimation (pinhole camera model)
    - Optional SAHI sliced inference for small/distant drones

Usage:
    python detect_drone_ultra.py                          # webcam
    python detect_drone_ultra.py --source video.mp4
    python detect_drone_ultra.py --imgsz 1280             # higher resolution
    python detect_drone_ultra.py --conf 0.4               # lower confidence
    python detect_drone_ultra.py --sahi                   # sliced inference
    python detect_drone_ultra.py --drone-width 0.5        # larger drone
"""

import argparse
import math
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = "best_new.pt"
MAX_BOX_RATIO = 0.6

COLOR_UNCONFIRMED = (0, 200, 255)  # yellow
COLOR_CONFIRMED = (0, 255, 0)      # green
COLOR_HUD = (0, 200, 255)          # orange


def estimate_focal_length(frame_width: int, fov_degrees: float) -> float:
    return frame_width / (2 * math.tan(math.radians(fov_degrees / 2)))


def estimate_distance(focal_length: float, real_width: float, pixel_width: int) -> float:
    if pixel_width <= 0:
        return -1
    return (real_width * focal_length) / pixel_width


def run_sahi(model_path: str, frame: np.ndarray, imgsz: int, conf: float,
             device: str) -> list[tuple]:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=320,
        slice_width=320,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0,
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        conf_score = pred.score.value
        detections.append((x1, y1, x2, y2, conf_score, None))
    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index, video, or image")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (pixels)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--drone-width", type=float, default=0.35,
                        help="Real drone width in meters for distance estimation (default: 0.35)")
    parser.add_argument("--min-frames", type=int, default=3,
                        help="Consecutive frames before confirming a track (default: 3)")
    parser.add_argument("--sahi", action="store_true",
                        help="Enable SAHI sliced inference for small/distant drones")
    parser.add_argument("--fov", type=float, default=70,
                        help="Camera horizontal FOV in degrees (default: 70)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Tracking: ByteTrack | Temporal filter: {args.min_frames} frames")
    print(f"Distance estimation: drone width={args.drone_width}m, FOV={args.fov}deg")
    if args.sahi:
        print("SAHI: enabled (sliced inference)")

    model = YOLO(MODEL_PATH)

    source = int(args.source) if args.source.isdigit() else args.source
    is_image = isinstance(source, str) and source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        results = model.predict(source, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device)
        frame = results[0].plot()
        cv2.imshow("Drone Detection", frame)
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(source)
    prev = time.time()

    # Tracking state: track_id -> consecutive frame count
    track_frames: dict[int, int] = {}
    confirmed_ids: set[int] = set()
    focal_length = None
    sahi_model = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        if focal_length is None:
            focal_length = estimate_focal_length(fw, args.fov)

        # --- Detection ---
        if args.sahi:
            # SAHI: sliced inference (no tracking integration)
            raw_detections = run_sahi(MODEL_PATH, frame, args.imgsz, args.conf, device)
        else:
            # Standard tracking inference
            results = model.track(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                  device=device, verbose=False, persist=True, tracker="bytetrack.yaml")
            raw_detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf_score = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                raw_detections.append((x1, y1, x2, y2, conf_score, track_id))

        # --- Filter oversized boxes ---
        detections = []
        for det in raw_detections:
            x1, y1, x2, y2, conf_score, track_id = det
            bw, bh = x2 - x1, y2 - y1
            if bw / fw > MAX_BOX_RATIO or bh / fh > MAX_BOX_RATIO:
                continue
            detections.append(det)

        # --- Temporal filtering ---
        seen_ids = set()
        for x1, y1, x2, y2, conf_score, track_id in detections:
            if track_id is not None:
                seen_ids.add(track_id)
                track_frames[track_id] = track_frames.get(track_id, 0) + 1

                # Log new confirmed drone
                if track_id not in confirmed_ids and track_frames[track_id] >= args.min_frames:
                    confirmed_ids.add(track_id)
                    print(f"[CONFIRMED] drone #{track_id} (seen {track_frames[track_id]} frames)")

        # Remove stale tracks
        lost_ids = set(track_frames.keys()) - seen_ids
        for tid in lost_ids:
            if tid in confirmed_ids:
                print(f"[LOST] drone #{tid}")
                confirmed_ids.discard(tid)
            del track_frames[tid]

        # --- Draw ---
        confirmed_count = 0
        for x1, y1, x2, y2, conf_score, track_id in detections:
            bw = x2 - x1
            dist = estimate_distance(focal_length, args.drone_width, bw)

            if track_id is not None and track_id in confirmed_ids:
                color = COLOR_CONFIRMED
                confirmed_count += 1
                dist_str = f" ~{dist:.0f}m" if dist > 0 else ""
                label = f"drone #{track_id} {conf_score:.2f}{dist_str}"
            elif track_id is not None and track_frames.get(track_id, 0) < args.min_frames:
                color = COLOR_UNCONFIRMED
                label = f"? {conf_score:.2f}"
            else:
                # SAHI mode (no tracking) or no ID
                color = COLOR_CONFIRMED
                confirmed_count += 1
                dist_str = f" ~{dist:.0f}m" if dist > 0 else ""
                label = f"drone {conf_score:.2f}{dist_str}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- HUD ---
        now = time.time()
        fps = 1 / (now - prev)
        prev = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_HUD, 2)
        cv2.putText(frame, f"Drones: {confirmed_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_HUD, 2)
        mode = "SAHI" if args.sahi else "Track"
        cv2.putText(frame, mode, (fw - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_HUD, 2)

        cv2.imshow("Drone Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
