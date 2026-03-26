"""Debug OAK blob raw output — prints shape, range, and top detections.

Usage:
    python tools/debug_oak_output.py
    python tools/debug_oak_output.py --blob models/nuno_yolov8n.blob
"""

import argparse
import time

import depthai as dai
import numpy as np

parser = argparse.ArgumentParser(description="Inspect raw OAK blob output")
parser.add_argument("--blob", default="models/best_yolov6n_openvino_2022.1_6shave.blob")
parser.add_argument("--img-size", type=int, default=640)
args = parser.parse_args()

print(f"Blob: {args.blob}")
print(f"Input: {args.img_size}x{args.img_size}")

with dai.Pipeline() as p:
    cam = p.create(dai.node.Camera).build()
    co = cam.requestOutput((args.img_size, args.img_size), type=dai.ImgFrame.Type.BGR888p)
    nn = p.create(dai.node.NeuralNetwork)
    nn.setBlobPath(args.blob)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(True)
    co.link(nn.input)
    q = nn.out.createOutputQueue(maxSize=1, blocking=True)
    p.start()
    print("Waiting for inference...\n")
    time.sleep(2)
    d = q.get()
    t = np.array(d.getFirstTensor())

    print(f"Raw shape: {t.shape}  dtype: {t.dtype}")
    print(f"Min: {t.min():.6f}  Max: {t.max():.6f}")
    print()

    # Try to auto-detect format and print top detections
    if t.ndim == 3:
        print(f"3D tensor: {t.shape}")
        # Could be (1, 8400, 6) for v6 or (1, 5, 8400) for v8
        _, d1, d2 = t.shape

        if d2 <= 10:
            # Likely (1, N, cols) — e.g. (1, 8400, 6)
            r = t[0]  # (N, cols)
            print(f"  Interpreting as (1, {d1}, {d2}) — rows x cols")
            print(f"  Last col top 5 (likely class score):")
            top = r[r[:, -1].argsort()[::-1][:5]]
            for row in top:
                print(f"    {['%.2f' % v for v in row]}")
        elif d1 <= 10:
            # Likely (1, cols, N) — e.g. (1, 5, 8400)
            r = t[0].T  # transpose to (N, cols)
            print(f"  Interpreting as (1, {d1}, {d2}) — need transpose to ({d2}, {d1})")
            print(f"  Last col top 5 (likely score):")
            top = r[r[:, -1].argsort()[::-1][:5]]
            for row in top:
                print(f"    {['%.2f' % v for v in row]}")
        else:
            print(f"  Unexpected dimensions ({d1}, {d2}) — printing raw slices")
            print(f"  t[0, :3, :5] = {t[0, :3, :5]}")
    elif t.ndim == 2:
        print(f"2D tensor: {t.shape}")
        r = t
        print(f"  Last col top 5:")
        top = r[r[:, -1].argsort()[::-1][:5]]
        for row in top:
            print(f"    {['%.2f' % v for v in row]}")
    else:
        print(f"Unexpected ndim={t.ndim}, printing flat stats")
        print(f"  First 20 values: {t.flatten()[:20]}")

    print(f"\nDone.")
