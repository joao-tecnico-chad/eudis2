"""Abstract base for drone detectors."""

from abc import ABC, abstractmethod

import numpy as np

from guardian.utils.decode import Detection


class DetectorABC(ABC):
    @abstractmethod
    def start(self) -> None:
        """Initialize the detection pipeline."""

    @abstractmethod
    def get_frame_and_detections(self) -> tuple[np.ndarray | None, list[Detection]]:
        """Get the latest frame and detections.

        Returns (frame, detections) where frame may be None if not ready.
        """

    def get_jpeg(self) -> bytes | None:
        """Get hardware-encoded JPEG frame as bytes. Override for HW encoding."""
        return None

    @abstractmethod
    def stop(self) -> None:
        """Release resources."""
