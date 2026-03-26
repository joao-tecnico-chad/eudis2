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
parser.add_argument("--hold", type=float, default=3.0, help="Seconds of continuous detection before confirming drone")
args = parser.parse_args()

BLOB_PATH = str(Path(args.blob).resolve())
IMG_SIZE = args.img_size
CONF_THRESH = args.conf
IOU_THRESH = args.iou
NUM_CLASSES = 1
LABELS = ["drone"]

HOLD_SEC = args.hold

print(f"Blob:  {BLOB_PATH}")
print(f"Input: {IMG_SIZE}x{IMG_SIZE}  conf>{CONF_THRESH}  iou>{IOU_THRESH}  hold>{HOLD_SEC}s")

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

    fps_time = time.time()
    frame_count = 0
    last_status = ""

    # Tracking state — follow a single detection spatially
    track_cx, track_cy = 0.0, 0.0  # tracked centroid (pixels)
    detect_start = None   # when continuous tracking began
    last_detect = None    # last time tracked target was seen
    confirmed = False
    GAP_TOLERANCE = 0.5   # seconds without match before resetting
    MATCH_DIST = IMG_SIZE * 0.25  # max pixel distance to count as same target
    CONFIRM_CONF = CONF_THRESH * 1.5  # min confidence to maintain confirmed track
    peak_conf = 0.0  # best confidence seen during this track

    try:
        while pipeline.isRunning():
            in_nn = q_nn.tryGet()
            if in_nn is None:
                time.sleep(0.001)
                continue

            output = np.array(in_nn.getFirstTensor()).reshape(-1, 6)
            detections = decode_yolov6(output, IMG_SIZE, NUM_CLASSES, CONF_THRESH, IOU_THRESH)

            frame_count += 1
            now = time.time()
            elapsed = now - fps_time

            if detections:
                # Find the detection closest to tracked position (or best if no track)
                if detect_start is not None:
                    # Match nearest to tracked centroid
                    def dist_to_track(d):
                        cx = (d.x1 + d.x2) / 2.0
                        cy = (d.y1 + d.y2) / 2.0
                        return ((cx - track_cx)**2 + (cy - track_cy)**2) ** 0.5
                    nearest = min(detections, key=dist_to_track)
                    if dist_to_track(nearest) < MATCH_DIST:
                        best = nearest
                    else:
                        # Nothing near the tracked target — treat as no match
                        best = None
                else:
                    # No active track — pick highest confidence to start
                    best = max(detections, key=lambda d: d.confidence)

                if best is not None:
                    # If confidence dropped well below peak, drone likely gone
                    if confirmed and peak_conf > 0 and best.confidence < peak_conf * 0.4:
                        print(f"\r  drone lost (conf dropped {peak_conf:.0%} -> {best.confidence:.0%})")
                        detect_start = None
                        last_detect = None
                        confirmed = False
                        peak_conf = 0.0
                        status = "\r  no drone       "
                    else:
                        # Update tracked centroid
                        track_cx = (best.x1 + best.x2) / 2.0
                        track_cy = (best.y1 + best.y2) / 2.0
                        if detect_start is None:
                            detect_start = now
                        last_detect = now
                        peak_conf = max(peak_conf, best.confidence)
                        hold_time = now - detect_start
                        w = best.x2 - best.x1
                        h = best.y2 - best.y1

                        if hold_time >= HOLD_SEC:
                            if not confirmed:
                                confirmed = True
                                print()
                            status = (
                                f"\r>>> DRONE CONFIRMED  "
                                f"best={best.confidence:.0%}  "
                                f"box={w}x{h}px  "
                                f"pos=({int(track_cx)},{int(track_cy)})  "
                                f"hold={hold_time:.0f}s"
                            )
                        else:
                            status = (
                                f"\r  ? tracking...  "
                                f"{hold_time:.1f}/{HOLD_SEC:.0f}s  "
                                f"best={best.confidence:.0%}  "
                                f"pos=({int(track_cx)},{int(track_cy)})"
                            )
                else:
                    status = f"\r  noise ({len(detections)} far from track)"
            else:
                status = "\r  no drone       "

            # Reset track if gap exceeded
            if last_detect is not None and (now - last_detect) > GAP_TOLERANCE:
                if confirmed:
                    print(f"\r  drone lost                                       ")
                detect_start = None
                last_detect = None
                confirmed = False
                peak_conf = 0.0

            # Update FPS every 2 seconds
            if elapsed >= 2.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = now
                status += f"  [{fps:.1f} fps]"

            # Overwrite line in-place
            padded = status.ljust(70)
            if padded != last_status:
                print(padded, end="", flush=True)
                last_status = padded

    except KeyboardInterrupt:
        print()

print("Done.")
