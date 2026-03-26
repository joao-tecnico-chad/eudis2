"""Debug OAK blob raw output — prints shape, range, and top detections."""

import time

import depthai as dai
import numpy as np

with dai.Pipeline() as p:
    cam = p.create(dai.node.Camera).build()
    co = cam.requestOutput((640, 640), type=dai.ImgFrame.Type.BGR888p)
    nn = p.create(dai.node.NeuralNetwork)
    nn.setBlobPath("models/best_yolov6n_openvino_2022.1_6shave.blob")
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(True)
    co.link(nn.input)
    q = nn.out.createOutputQueue(maxSize=1, blocking=True)
    p.start()
    time.sleep(2)
    d = q.get()
    t = np.array(d.getFirstTensor())
    print(f"shape: {t.shape}  dtype: {t.dtype}")
    print(f"min: {t.min():.4f}  max: {t.max():.4f}")
    r = t.reshape(-1, 6)
    top = r[r[:, 5].argsort()[::-1][:5]]
    print("Top 5 by class score:")
    for row in top:
        print(f"  cx={row[0]:.1f} cy={row[1]:.1f} w={row[2]:.1f} h={row[3]:.1f} obj={row[4]:.4f} cls={row[5]:.4f}")
