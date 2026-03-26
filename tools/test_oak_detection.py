"""Live detection test on OAK-1W with YOLOv6n blob (depthai v3) — terminal only.

Usage (on Pi Zero):
    python tools/test_oak_detection.py
    python tools/test_oak_detection.py --conf 0.3
"""

import argparse
import sys
import time
from pathlib import Path

import depthai as dai
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from guardian.utils.decode import decode_yolov6

parser = argparse.ArgumentParser(description="OAK-1W YOLOv6n live detection")
parser.add_argument("--blob", default="models/best_yolov6n_openvino_2022.1_6shave.blob")
parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
parser.add_argument("--img-size", type=int, default=640)
args = parser.parse_args()

BLOB_PATH = str(Path(args.blob).resolve())
IMG_SIZE = args.img_size
CONF_THRESH = args.conf
IOU_THRESH = args.iou
NUM_CLASSES = 1
LABELS = ["drone"]

print(f"Blob:  {BLOB_PATH}")
print(f"Input: {IMG_SIZE}x{IMG_SIZE}  conf>{CONF_THRESH}  iou>{IOU_THRESH}")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((IMG_SIZE, IMG_SIZE), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumInferenceThreads(1)
    nn.input.setBlocking(True)
    cam_out.link(nn.input)

    q_nn = nn.out.createOutputQueue(maxSize=1, blocking=True)

    pipeline.start()
    print("Running — Ctrl+C to quit\n")

    fps_time = time.time()
    frame_count = 0
    last_status = ""

    try:
        while pipeline.isRunning():
            in_nn = q_nn.tryGet()
            if in_nn is None:
                time.sleep(0.001)
                continue

            output = np.array(in_nn.getFirstTensor()).reshape(-1, 6)
            detections = decode_yolov6(output, IMG_SIZE, NUM_CLASSES, CONF_THRESH, IOU_THRESH)

            frame_count += 1
            elapsed = time.time() - fps_time

            if detections:
                best = max(detections, key=lambda d: d.confidence)
                w = best.x2 - best.x1
                h = best.y2 - best.y1
                status = (
                    f"\r DRONE DETECTED  "
                    f"n={len(detections)}  "
                    f"best={best.confidence:.0%}  "
                    f"box={w}x{h}px"
                )
            else:
                status = "\r  no drone       "

            # Update FPS every 2 seconds
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()
                status += f"  [{fps:.1f} fps]"

            # Overwrite line in-place
            padded = status.ljust(60)
            if detections or padded != last_status:
                print(padded, end="", flush=True)
                last_status = padded

    except KeyboardInterrupt:
        print()

print("Done.")
