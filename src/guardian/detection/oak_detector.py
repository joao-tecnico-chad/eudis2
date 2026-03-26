"""DepthAI OAK-1W detector — runs YOLOv6/v8 blob on the MyriadX VPU."""

from pathlib import Path

import numpy as np

from guardian.config import GuardianConfig
from guardian.detection.base import DetectorABC
from guardian.utils.decode import Detection, decode_yolov6, decode_yolov8


class OakDetector(DetectorABC):
    def __init__(self, config: GuardianConfig):
        self._config = config
        self._pipeline = None
        self._q_rgb = None
        self._q_nn = None

    def start(self) -> None:
        import depthai as dai

        blob_path = Path(self._config.blob_path)
        if not blob_path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_path}")

        self._pipeline = dai.Pipeline()

        cam = self._pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            (self._config.img_size, self._config.img_size),
            dai.ImgFrame.Type.BGR888p,
        )

        nn = self._pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(blob_path))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(True)
        cam_out.link(nn.input)

        self._q_rgb = cam_out.createOutputQueue(maxSize=4, blocking=False)
        self._q_nn = nn.out.createOutputQueue(maxSize=4, blocking=False)

        self._pipeline.start()
        print(f"OAK-1W detector started ({self._config.model_format}, "
              f"{self._config.img_size}x{self._config.img_size})")

    def get_frame_and_detections(self) -> tuple[np.ndarray | None, list[Detection]]:
        in_rgb = self._q_rgb.tryGet()
        in_nn = self._q_nn.tryGet()

        if in_rgb is None:
            return None, []

        frame = in_rgb.getCvFrame()

        if in_nn is None:
            return frame, []

        output = np.array(in_nn.getFirstTensor())

        if self._config.model_format == "yolov8":
            detections = decode_yolov8(
                output, scale=1.0,
                conf_thresh=self._config.conf_threshold,
                iou_thresh=self._config.iou_threshold,
            )
        else:
            detections = decode_yolov6(
                output, self._config.img_size, self._config.num_classes,
                self._config.conf_threshold, self._config.iou_threshold,
            )

        # Filter oversized boxes (likely false positives)
        fh, fw = frame.shape[:2]
        detections = [
            d for d in detections
            if (d.x2 - d.x1) / fw <= self._config.max_box_ratio
            and (d.y2 - d.y1) / fh <= self._config.max_box_ratio
        ]

        return frame, detections

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        print("OAK-1W detector stopped")
