"""Drone Guardian main loop."""

import time

from guardian.activation.filter import ActivationFilter, ActivationState
from guardian.config import GuardianConfig
from guardian.streaming.server import StreamServer
from guardian.utils.decode import Detection


class DroneGuardian:
    def __init__(self, config: GuardianConfig, detect_only: bool = False):
        self.config = config
        self.detect_only = detect_only
        self.detector = self._create_detector()
        self.activation = ActivationFilter(config)
        self.stream_server = StreamServer(config)

    def _create_detector(self):
        if self.config.is_pi():
            from guardian.detection.oak_detector import OakDetector
            return OakDetector(self.config)
        else:
            from guardian.detection.stub_detector import StubDetector
            return StubDetector(self.config)

    def run(self) -> None:
        print("=== Drone Guardian starting ===")
        self.detector.start()
        self.stream_server.start()

        prev_time = time.monotonic()
        fps = 0.0
        is_pi = self.config.is_pi()

        try:
            while True:
                # Push video (hardware MJPEG on Pi)
                if is_pi:
                    jpeg = self.detector.get_jpeg()
                    if jpeg:
                        self.stream_server.push_jpeg(jpeg)

                # Get NN detections
                _, detections = self.detector.get_frame_and_detections()

                if detections is None:
                    time.sleep(0.001)
                    continue

                # Desktop: push frame for CPU encoding
                if not is_pi:
                    frame, _ = self.detector.get_frame_and_detections()
                    if frame is not None:
                        self.stream_server.push_frame(frame)

                # FPS (per NN inference)
                now = time.monotonic()
                dt = now - prev_time
                if dt > 0:
                    fps = 1.0 / dt
                prev_time = now

                if detections:
                    best = max(detections, key=lambda d: d.confidence)
                    print(f"\r  DETECT n={len(detections)} best={best.confidence:.0%} fps={fps:.0f}   ", end="", flush=True)

                # Push detections to dashboard
                state = ActivationState()
                self.stream_server.push_telemetry(0.0, detections, state, fps)

        except KeyboardInterrupt:
            print("\nShutdown requested")
        finally:
            self.detector.stop()
            self.stream_server.stop()
            print("=== Drone Guardian stopped ===")
