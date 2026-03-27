"""OAK-1W detector using DetectionNetwork + NNArchive.

All YOLO decoding happens on the MyriadX VPU.
Hardware MJPEG encoder for video streaming.
"""

from pathlib import Path

import numpy as np

from guardian.config import GuardianConfig
from guardian.detection.base import DetectorABC
from guardian.utils.decode import Detection


class OakDetector(DetectorABC):
    def __init__(self, config: GuardianConfig):
        self._config = config
        self._pipeline = None
        self._q_det = None
        self._q_mjpeg = None
        self._labels = []

    def start(self) -> None:
        import depthai as dai

        model_path = Path(self._config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._pipeline = dai.Pipeline()
        self._pipeline.setXLinkChunkSize(0)

        cam = self._pipeline.create(dai.node.Camera).build()

        # Camera at high res, ImageManip resizes for NN (better quality)
        cam_preview = cam.requestOutput(
            (1280, 720), dai.ImgFrame.Type.BGR888p, fps=10,
        )

        nn_archive = dai.NNArchive(str(model_path))
        nn_size = nn_archive.getInputSize()

        manip = self._pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setOutputSize(nn_size[0], nn_size[1])
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
        manip.setMaxOutputFrameSize(nn_size[0] * nn_size[1] * 3)
        cam_preview.link(manip.inputImage)

        # DetectionNetwork — on-device YOLO decode
        det_nn = self._pipeline.create(dai.node.DetectionNetwork).build(
            manip.out, nn_archive
        )
        det_nn.setConfidenceThreshold(self._config.conf_threshold)
        det_nn.setNumInferenceThreads(2)
        self._labels = det_nn.getClasses()

        # Hardware MJPEG encoder
        enc_out = cam.requestOutput(
            (640, 480), dai.ImgFrame.Type.NV12, fps=10,
        )
        encoder = self._pipeline.create(dai.node.VideoEncoder)
        encoder.setDefaultProfilePreset(
            10, dai.VideoEncoderProperties.Profile.MJPEG,
        )
        encoder.setQuality(self._config.jpeg_quality)
        enc_out.link(encoder.input)

        self._q_det = det_nn.out.createOutputQueue(maxSize=1, blocking=False)
        self._q_mjpeg = encoder.out.createOutputQueue(maxSize=1, blocking=False)

        self._pipeline.start()
        print(f"OAK-1W started (DetectionNetwork, {nn_size[0]}x{nn_size[1]}, HW MJPEG)")

    def get_frame_and_detections(self) -> tuple[np.ndarray | None, list[Detection] | None]:
        msg = self._q_det.tryGet()
        if msg is None:
            return None, None

        detections = []
        for d in msg.detections:
            label = self._labels[d.label] if d.label < len(self._labels) else "drone"
            # Convert normalized coords (0-1) to pixel coords
            img = self._config.img_size
            detections.append(Detection(
                x1=int(max(0, d.xmin) * img),
                y1=int(max(0, d.ymin) * img),
                x2=int(min(1, d.xmax) * img),
                y2=int(min(1, d.ymax) * img),
                confidence=d.confidence,
                class_id=d.label,
            ))

        return None, detections

    def get_jpeg(self) -> bytes | None:
        msg = self._q_mjpeg.tryGet()
        if msg is None:
            return None
        return bytes(msg.getData())

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        print("OAK-1W detector stopped")
