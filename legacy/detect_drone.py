"""
Drone Detection using YOLOv8 (colleague's model, Zhejiang dataset)

Usage:
    python detect_drone.py --source 0              # webcam
    python detect_drone.py --source video.mp4      # video file
    python detect_drone.py --source image.jpg      # image file
    python detect_drone.py --source rtsp://...     # RTSP stream
    python detect_drone.py --track                 # enable object tracking
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = Path("best.onnx")


def run(source: str, conf: float, iou: float, save: bool, show: bool, track: bool) -> None:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model = YOLO(str(MODEL_PATH), task="detect")

    kwargs = dict(source=source, conf=conf, iou=iou, save=save, show=show, stream=True, verbose=True, device=device)

    results = model.track(**kwargs) if track else model.predict(**kwargs)

    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                confidence = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                track_id = int(box.id[0]) if track and box.id is not None else None
                id_str = f" | ID: {track_id}" if track_id is not None else ""
                print(f"drone | Confidence: {confidence:.2f}{id_str} | Box: {[round(c, 1) for c in coords]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8x Drone Detector")
    parser.add_argument("--source", type=str, default="0", help="Input source (webcam index, file path, or stream URL)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--save", action="store_true", help="Save annotated results to runs/detect/")
    parser.add_argument("--track", action="store_true", help="Enable object tracking (assigns IDs to drones)")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable live preview window")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    run(source=args.source, conf=args.conf, iou=args.iou, save=args.save, show=args.show, track=args.track)


if __name__ == "__main__":
    main()
