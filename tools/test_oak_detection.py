"""Live detection on OAK-1W with terminal output and single-target tracking.

Usage:
    python tools/test_oak_detection.py
    python tools/test_oak_detection.py --model-format yolov8 --conf 0.5
    python tools/test_oak_detection.py --blob models/custom.blob --conf 0.4
"""

import argparse
import sys
import time
from pathlib import Path

import depthai as dai
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from guardian.utils.decode import decode_yolov6, decode_yolov8
from guardian.utils.tracker import DetectionTracker

DEFAULT_BLOBS = {
    "yolov6": "models/best_yolov6n_openvino_2022.1_6shave.blob",
    "yolov8": "models/nuno_yolov8n.blob",
}

parser = argparse.ArgumentParser(description="OAK-1W live drone detection")
parser.add_argument("--model-format", choices=["yolov6", "yolov8"], default="yolov6")
parser.add_argument("--blob", default=None, help="Path to .blob (auto-selects from model-format)")
parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
parser.add_argument("--img-size", type=int, default=640)
parser.add_argument("--hold", type=float, default=3.0, help="Seconds to confirm drone")
parser.add_argument("--max-box-ratio", type=float, default=0.6, help="Max box size as fraction of frame")
args = parser.parse_args()

BLOB_PATH = str(Path(args.blob or DEFAULT_BLOBS[args.model_format]).resolve())
IMG_SIZE = args.img_size
CONF = args.conf
IOU = args.iou
NUM_CLASSES = 1

print(f"Model:  {args.model_format}")
print(f"Blob:   {BLOB_PATH}")
print(f"Input:  {IMG_SIZE}x{IMG_SIZE}  conf>{CONF}  iou>{IOU}  hold>{args.hold}s")


def decode(raw: np.ndarray) -> list:
    if args.model_format == "yolov8":
        return decode_yolov8(raw, scale=1.0, conf_thresh=CONF, iou_thresh=IOU)
    return decode_yolov6(raw, IMG_SIZE, NUM_CLASSES, CONF, IOU)


def format_status(state, fps_str: str) -> str:
    if state.lost_reason:
        return f"\r  drone lost ({state.lost_reason})"
    if state.confirmed:
        d = state.best_det
        w, h = d.x2 - d.x1, d.y2 - d.y1
        return (
            f"\r{fps_str}>>> DRONE  "
            f"hold={state.hold_time:.0f}s  "
            f"conf={d.confidence:.0%}  "
            f"ema={state.ema_conf:.0%}  "
            f"box={w}x{h}  "
            f"pos=({int(state.cx)},{int(state.cy)})"
        )
    if state.tracking:
        return (
            f"\r{fps_str}? tracking "
            f"{state.hold_time:.1f}/{args.hold:.0f}s  "
            f"conf={state.best_det.confidence:.0%}  "
            f"ema={state.ema_conf:.0%}  "
            f"pos=({int(state.cx)},{int(state.cy)})"
        )
    return f"\r{fps_str}  no drone"


# --- Pipeline setup ---
with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((IMG_SIZE, IMG_SIZE), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(True)
    cam_out.link(nn.input)

    q_nn = nn.out.createOutputQueue(maxSize=1, blocking=True)

    pipeline.start()
    print("Running — Ctrl+C to quit\n")

    tracker = DetectionTracker(
        img_size=IMG_SIZE,
        hold_sec=args.hold,
        max_box_ratio=args.max_box_ratio,
    )

    fps_time = time.time()
    frame_count = 0
    fps_str = ""
    last_line = ""

    try:
        while pipeline.isRunning():
            in_nn = q_nn.tryGet()
            if in_nn is None:
                time.sleep(0.001)
                continue

            raw = np.array(in_nn.getFirstTensor())
            detections = decode(raw)
            state = tracker.update(detections, time.time())

            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 2.0:
                fps_str = f"[{frame_count / elapsed:.1f} fps]  "
                frame_count = 0
                fps_time = time.time()

            line = format_status(state, fps_str).ljust(80)

            if state.lost_reason:
                print(line)
                last_line = ""
            elif line != last_line:
                print(line, end="", flush=True)
                last_line = line

    except KeyboardInterrupt:
        print()

print("Done.")
