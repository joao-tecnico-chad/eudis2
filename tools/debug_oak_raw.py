"""Deep inspection of OAK blob raw output values."""

import argparse
import time

import depthai as dai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--blob", default="models/drone_yolov6n.blob")
parser.add_argument("--img-size", type=int, default=416)
args = parser.parse_args()

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
    time.sleep(2)
    d = q.get()
    t = np.array(d.getFirstTensor())

    print(f"Raw shape: {t.shape}  dtype: {t.dtype}")
    r = t.reshape(-1, t.shape[-1])
    ncols = r.shape[1]
    print(f"Reshaped: {r.shape}")
    print()

    for c in range(ncols):
        col = r[:, c]
        print(f"Col {c}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}  std={col.std():.4f}")
    print()

    # Check if sigmoid is needed on columns 4 and 5
    from scipy.special import expit as sigmoid
    print("After sigmoid on col 4 and 5:")
    for c in [4, 5]:
        col = sigmoid(r[:, c])
        print(f"  Col {c}: min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")
        for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            print(f"    >{thresh}: {(col > thresh).sum()}")
    print()

    print("Top 10 by col 5 (raw):")
    top = r[r[:, 5].argsort()[::-1][:10]]
    for row in top:
        print(f"  cx={row[0]:.1f} cy={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} obj={row[4]:.4f} cls={row[5]:.4f}")

    print()
    print("Top 10 by sigmoid(col 5):")
    sig5 = sigmoid(r[:, 5])
    top_idx = sig5.argsort()[::-1][:10]
    for i in top_idx:
        row = r[i]
        print(f"  cx={row[0]:.1f} cy={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} obj={row[4]:.4f} cls_raw={row[5]:.4f} cls_sig={sig5[i]:.4f}")
