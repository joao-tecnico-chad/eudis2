"""DepthAI OAK-1W detector — runs YOLOv6/v8 blob on the MyriadX VPU.

Uses hardware VideoEncoder for MJPEG streaming (zero Pi CPU for video encoding).
"""

from pathlib import Path

import numpy as np

from guardian.config import GuardianConfig
from guardian.detection.base import DetectorABC
from guardian.utils.decode import Detection, decode_yolov6, decode_yolov8


class OakDetector(DetectorABC):
    def __init__(self, config: GuardianConfig):
        self._config = config
        self._pipeline = None
        self._q_nn = None
        self._q_mjpeg = None

    def start(self) -> None:
        import depthai as dai

        blob_path = Path(self._config.blob_path)
        if not blob_path.exists():
            raise FileNotFoundError(f"Blob not found: {blob_path}")

        self._pipeline = dai.Pipeline()

        # Camera
        cam = self._pipeline.create(dai.node.Camera).build()

        # NN input — model resolution (e.g. 416x416)
        nn_out = cam.requestOutput(
            (self._config.img_size, self._config.img_size),
            dai.ImgFrame.Type.BGR888p,
        )

        # Stream output — higher res for viewing, hardware MJPEG encoded
        stream_out = cam.requestOutput(
            (640, 480),
            dai.ImgFrame.Type.NV12,
        )
        encoder = self._pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(
            15, dai.VideoEncoderProperties.Profile.MJPEG
        )
        encoder.setQuality(self._config.jpeg_quality)
        stream_out.link(encoder.input)

        # Neural network
        nn = self._pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(blob_path))
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(True)
        nn_out.link(nn.input)

        # Output queues
        self._q_nn = nn.out.createOutputQueue(maxSize=4, blocking=False)
        self._q_mjpeg = encoder.out.createOutputQueue(maxSize=4, blocking=False)

        self._pipeline.start()
        print(f"OAK-1W detector started ({self._config.model_format}, "
              f"{self._config.img_size}x{self._config.img_size}, HW MJPEG)")

    def get_frame_and_detections(self) -> tuple[np.ndarray | None, list[Detection]]:
        in_nn = self._q_nn.tryGet()

        if in_nn is None:
            return None, []

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

        # Filter oversized boxes
        img = self._config.img_size
        detections = [
            d for d in detections
            if (d.x2 - d.x1) / img <= self._config.max_box_ratio
            and (d.y2 - d.y1) / img <= self._config.max_box_ratio
        ]

        return None, detections  # no numpy frame needed — MJPEG is separate

    def get_jpeg(self) -> bytes | None:
        """Get hardware-encoded MJPEG frame as raw bytes. Zero CPU cost."""
        in_mjpeg = self._q_mjpeg.tryGet()
        if in_mjpeg is None:
            return None
        return bytes(in_mjpeg.getData())

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        print("OAK-1W detector stopped")
