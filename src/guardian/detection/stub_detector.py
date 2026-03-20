"""Desktop stub detector — webcam/video via OpenCV + optional ONNX inference."""

from pathlib import Path

import cv2
import numpy as np

from guardian.config import GuardianConfig
from guardian.detection.base import DetectorABC
from guardian.utils.decode import Detection, decode_yolov8, preprocess_frame


class StubDetector(DetectorABC):
    """Desktop detector using OpenCV capture and optional ONNX model."""

    def __init__(self, config: GuardianConfig):
        self._config = config
        self._cap = None
        self._session = None
        self._input_name = None
        self._onnx_img_size: int | None = None

    def start(self) -> None:
        source = self._config.source
        self._cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Try to load ONNX model for real inference
        onnx_path = Path(self._config.onnx_path)
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                preferred = [p for p in ["CoreMLExecutionProvider", "CUDAExecutionProvider",
                                         "CPUExecutionProvider"] if p in providers]
                self._session = ort.InferenceSession(str(onnx_path), providers=preferred)
                self._input_name = self._session.get_inputs()[0].name
                # Read expected input size from model (e.g., 640 for YOLOv8)
                input_shape = self._session.get_inputs()[0].shape
                self._onnx_img_size = input_shape[2] if len(input_shape) >= 3 else None
                print(f"Stub detector: ONNX model loaded ({preferred[0]}, input={self._onnx_img_size}px)")
            except ImportError:
                print("Stub detector: onnxruntime not installed, detections disabled")
        else:
            print(f"Stub detector: ONNX model not found at {onnx_path}, detections disabled")

        print(f"Stub detector started (source: {source})")

    def get_frame_and_detections(self) -> tuple[np.ndarray | None, list[Detection]]:
        if self._cap is None:
            return None, []

        ret, frame = self._cap.read()
        if not ret:
            return None, []

        detections = []
        if self._session is not None:
            img_size = self._onnx_img_size or self._config.img_size
            blob, scale = preprocess_frame(frame, img_size)
            output = self._session.run(None, {self._input_name: blob})[0]
            detections = decode_yolov8(
                output, scale,
                self._config.conf_threshold, self._config.iou_threshold
            )
            # Filter oversized boxes
            fh, fw = frame.shape[:2]
            detections = [
                d for d in detections
                if (d.x2 - d.x1) / fw <= self._config.max_box_ratio
                and (d.y2 - d.y1) / fh <= self._config.max_box_ratio
            ]

        return frame, detections

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._session = None
        print("Stub detector stopped")
