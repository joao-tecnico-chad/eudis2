"""Live detection test on OAK-1W with YOLOv6n blob (depthai v3) — terminal only."""

import sys
import time
from pathlib import Path

import depthai as dai
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from guardian.utils.decode import decode_yolov6

BLOB_PATH = str(Path("models/best_yolov6n_openvino_2022.1_6shave.blob").resolve())
IMG_SIZE = 640
CONF_THRESH = 0.5
IOU_THRESH = 0.5
NUM_CLASSES = 1
LABELS = ["drone"]

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    cam_out = cam.requestOutput((IMG_SIZE, IMG_SIZE), type=dai.ImgFrame.Type.BGR888p)

    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(BLOB_PATH)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    cam_out.link(nn.input)

    q_nn = nn.out.createOutputQueue(maxSize=1, blocking=False)

    pipeline.start()
    print("Running YOLOv6n detection — Ctrl+C to quit")

    fps_time = time.time()
    frame_count = 0

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
                print(f"  [{len(detections)}] {[(LABELS[d.class_id], f'{d.confidence:.2f}') for d in detections]}")

            if elapsed >= 2.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()
                print(f"  --- FPS: {fps:.1f} ---")
    except KeyboardInterrupt:
        pass

print("Done.")
