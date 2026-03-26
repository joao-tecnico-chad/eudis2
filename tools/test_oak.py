"""Quick smoke test for OAK-1W camera (depthai v3)."""

import depthai as dai
import cv2

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    out = cam.requestOutput((416, 416))
    q = out.createOutputQueue()

    pipeline.start()

    frame = q.get()
    cv2.imwrite("/tmp/oak_test.jpg", frame.getCvFrame())
    print(f"Captured: {frame.getCvFrame().shape} -> /tmp/oak_test.jpg")
