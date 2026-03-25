"""Drone Guardian main loop orchestrator."""

import time

import cv2
import numpy as np

from guardian.activation.filter import ActivationFilter, ActivationState
from guardian.actuators.base import ServoABC
from guardian.config import GuardianConfig
from guardian.detection.base import DetectorABC
from guardian.sensors.base import BarometerABC
from guardian.streaming.server import StreamServer
from guardian.utils.decode import Detection
from guardian.utils.geometry import (
    box_centroid,
    estimate_distance,
    estimate_focal_length,
)


# HUD colors (BGR)
COLOR_BOX = (0, 255, 0)
COLOR_BOX_UNCONFIRMED = (0, 200, 255)
COLOR_HUD = (0, 200, 255)
COLOR_ARMED = (0, 255, 255)
COLOR_FIRED = (0, 0, 255)


class DroneGuardian:
    def __init__(self, config: GuardianConfig):
        self.config = config
        self.detector = self._create_detector()
        self.barometer = self._create_barometer()
        self.servo = self._create_servo()
        self.activation = ActivationFilter(config)
        self.stream_server = StreamServer(config)
        self._focal_length: float | None = None

    def _create_detector(self) -> DetectorABC:
        if self.config.is_pi():
            from guardian.detection.oak_detector import OakDetector
            return OakDetector(self.config)
        else:
            from guardian.detection.stub_detector import StubDetector
            return StubDetector(self.config)

    def _create_barometer(self) -> BarometerABC:
        if self.config.is_pi():
            from guardian.sensors.bmp390 import BMP390Barometer
            return BMP390Barometer()
        else:
            from guardian.sensors.stub_barometer import StubBarometer
            return StubBarometer()

    def _create_servo(self) -> ServoABC:
        if self.config.is_pi():
            from guardian.actuators.pwm_servo import PWMServo
            return PWMServo(self.config)
        else:
            from guardian.actuators.stub_servo import StubServo
            return StubServo()

    def run(self) -> None:
        print("=== Drone Guardian starting ===")
        print(f"Hardware mode: {'Pi' if self.config.is_pi() else 'Desktop'}")
        print(f"Activation: altitude>{self.config.altitude_margin_m}m, "
              f"zone={self.config.centroid_zone_ratio*100:.0f}%, "
              f"area>{self.config.min_box_area_ratio*100:.1f}%, "
              f"frames={self.config.consecutive_frames}")

        self.barometer.set_reference()
        self.detector.start()
        self.stream_server.start()

        prev_time = time.monotonic()
        fps = 0.0

        try:
            while True:
                # Read altitude first — skip inference if not airborne
                altitude_delta = self.barometer.get_altitude_delta_m()
                is_airborne = altitude_delta > self.config.altitude_margin_m

                frame, detections = self.detector.get_frame_and_detections()
                if frame is None:
                    time.sleep(0.001)
                    continue

                fh, fw = frame.shape[:2]
                if self._focal_length is None:
                    self._focal_length = estimate_focal_length(fw, self.config.camera_fov_deg)

                # Only keep detections if airborne — save compute on ground
                if not is_airborne:
                    detections = []

                # If servo finished rearming, clear cooldown
                if not self.servo.is_ready:
                    pass  # still cycling
                elif self.activation._cooling_down:
                    self.activation.mark_ready()
                    print("*** REARMED — ready to fire again ***")

                # Run activation filter
                state = self.activation.update(altitude_delta, detections, fw, fh)

                # Fire when armed and servo is ready
                if state.armed and self.servo.is_ready:
                    self.servo.fire()
                    self.activation.mark_fired()
                    print("*** NET DEPLOYED ***")

                # Compute FPS
                now = time.monotonic()
                dt = now - prev_time
                if dt > 0:
                    fps = 1.0 / dt
                prev_time = now

                # Draw HUD and push to stream
                annotated = self._draw_hud(frame, detections, state, altitude_delta, fps)
                self.stream_server.push_frame(annotated)
                self.stream_server.push_telemetry(altitude_delta, len(detections), state, fps)

        except KeyboardInterrupt:
            print("\nShutdown requested")
        finally:
            self.detector.stop()
            self.servo.safe()
            self.stream_server.stop()
            print("=== Drone Guardian stopped ===")

    def _draw_hud(self, frame: np.ndarray, detections: list[Detection],
                  state: ActivationState, altitude_delta: float, fps: float) -> np.ndarray:
        """Annotate frame with bounding boxes, activation status, and telemetry."""
        annotated = frame.copy()
        fh, fw = annotated.shape[:2]

        # Draw center zone rectangle
        zone = self.config.centroid_zone_ratio
        margin_x = (1.0 - zone) / 2.0
        margin_y = (1.0 - zone) / 2.0
        zx1, zy1 = int(margin_x * fw), int(margin_y * fh)
        zx2, zy2 = int((1 - margin_x) * fw), int((1 - margin_y) * fh)
        cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (80, 80, 80), 1)

        # Draw detections
        for det in detections:
            cx, cy = box_centroid(det.x1, det.y1, det.x2, det.y2)
            pixel_w = det.x2 - det.x1
            dist = estimate_distance(self._focal_length, self.config.real_drone_width_m, pixel_w)

            color = COLOR_BOX if state.layer4_count > 0 else COLOR_BOX_UNCONFIRMED
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            dist_str = f" ~{dist:.0f}m" if dist > 0 else ""
            label = f"drone {det.confidence:.2f}{dist_str}"
            cv2.putText(annotated, label, (det.x1, det.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw centroid
            cv2.circle(annotated, (int(cx), int(cy)), 4, color, -1)

        # Bottom-left: compact status bar
        bar_y = fh - 12
        status_items = [
            f"FPS:{fps:.0f}",
            f"ALT:{altitude_delta:.0f}m",
            f"L4:{state.layer4_count}/{self.config.consecutive_frames}",
        ]
        if state.cooling_down:
            status_items.append("REARMING")
        elif state.armed:
            status_items.append("ARMED")
        status_text = " | ".join(status_items)
        cv2.putText(annotated, status_text, (8, bar_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD, 1)

        return annotated
