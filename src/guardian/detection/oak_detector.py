"""DepthAI OAK-1 W detector — runs YOLOv6/v8 blob on the MyriadX VPU."""

from pathlib import Path

import numpy as np

from guardian.config import GuardianConfig
from guardian.detection.base import DetectorABC
from guardian.utils.decode import Detection, decode_yolov6


class OakDetector(DetectorABC):
    def __init__(self, config: GuardianConfig):
        self._config = config
        self._device = None
        self._q_rgb = None
        self._q_nn = None

    def start(self) -> None:
        import depthai as dai

        blob_path = Path(self._config.blob_path)
        if not blob_path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_path}")

        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(self._config.img_size, self._config.img_size)
        cam.setInterleaved(False)
        cam.setFp16(True)

        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(blob_path))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")

        cam.preview.link(nn.input)
        cam.preview.link(xout_rgb.input)
        nn.out.link(xout_nn.input)

        self._device = dai.Device(pipeline)
        self._q_rgb = self._device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self._q_nn = self._device.getOutputQueue("nn", maxSize=4, blocking=False)

        print("OAK-1 W detector started")

    def get_frame_and_detections(self) -> tuple[np.ndarray | None, list[Detection]]:
        in_rgb = self._q_rgb.tryGet()
        in_nn = self._q_nn.tryGet()

        if in_rgb is None:
            return None, []

        frame = in_rgb.getCvFrame()

        if in_nn is None:
            return frame, []

        output = np.array(in_nn.getFirstLayerFp16())

        if self._config.model_format == "yolov6":
            detections = decode_yolov6(
                output, self._config.img_size, self._config.num_classes,
                self._config.conf_threshold, self._config.iou_threshold
            )
        else:
            # YOLOv8 blob decoding would go here when blob format is finalized
            detections = decode_yolov6(
                output, self._config.img_size, self._config.num_classes,
                self._config.conf_threshold, self._config.iou_threshold
            )

        return frame, detections

    def stop(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
        print("OAK-1 W detector stopped")
