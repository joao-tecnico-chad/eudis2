"""
Drone Detection using YOLOv6 (locally trained weights)

Usage:
    python detect_drone_v6.py                          # webcam
    python detect_drone_v6.py --source video.mp4       # video file
    python detect_drone_v6.py --source image.jpg       # image
    python detect_drone_v6.py --model v6s              # use v6s weights
    python detect_drone_v6.py --save                   # save output video
"""

import argparse
import subprocess
import sys
from pathlib import Path

YOLOV6_DIR = Path("YOLOv6")

MODELS = {
    "v6n": YOLOV6_DIR / "runs" / "train" / "drone_yolov6n" / "weights" / "best_ckpt.pt",
    "v6s": YOLOV6_DIR / "runs" / "train" / "drone_yolov6s" / "weights" / "best_ckpt.pt",
}


def main():
    parser = argparse.ArgumentParser(description="YOLOv6 Drone Detector")
    parser.add_argument("--source", type=str, default="0",
                        help="Input: webcam index, file path, or stream URL")
    parser.add_argument("--model", choices=["v6n", "v6s"], default="v6n",
                        help="Model variant")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU threshold for NMS")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated output")
    parser.add_argument("--no-show", dest="show", action="store_false",
                        help="Disable live preview")
    parser.set_defaults(show=True)
    args = parser.parse_args()

    weights = MODELS[args.model]
    if not weights.exists():
        print(f"Weights not found: {weights}")
        print(f"Train first: python train.py --model {args.model}")
        sys.exit(1)

    cmd = [
        sys.executable, "tools/infer.py",
        "--weights", str(weights.resolve()),
        "--source", args.source,
        "--img-size", "416",
        "--conf-thres", str(args.conf),
        "--iou-thres", str(args.iou),
        "--device", "0",
    ]

    if not args.show:
        cmd.append("--hide-labels")

    if args.save:
        cmd.extend(["--save-dir", str(Path("runs/detect").resolve())])

    print(f"Running YOLOv6{args.model[1:]} inference on: {args.source}")
    subprocess.run(cmd, cwd=str(YOLOV6_DIR))


if __name__ == "__main__":
    main()
