"""Drone Guardian main loop orchestrator."""

import time

from guardian.activation.filter import ActivationFilter, ActivationState
from guardian.actuators.base import ServoABC
from guardian.config import GuardianConfig
from guardian.detection.base import DetectorABC
from guardian.sensors.base import BarometerABC
from guardian.streaming.server import StreamServer
from guardian.utils.decode import Detection


class DroneGuardian:
    def __init__(self, config: GuardianConfig, detect_only: bool = False):
        self.config = config
        self.detect_only = detect_only
        self.detector = self._create_detector()
        self.barometer = None if detect_only else self._create_barometer()
        self.servo = None if detect_only else self._create_servo()
        self.activation = ActivationFilter(config)
        self.stream_server = StreamServer(config)

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
        mode = "detect-only" if self.detect_only else ("Pi" if self.config.is_pi() else "Desktop")
        print(f"Mode: {mode}")
        if not self.detect_only:
            print(f"Activation: altitude>{self.config.altitude_margin_m}m, "
                  f"zone={self.config.centroid_zone_ratio*100:.0f}%, "
                  f"area>{self.config.min_box_area_ratio*100:.1f}%, "
                  f"frames={self.config.consecutive_frames}")

        if self.barometer:
            self.barometer.set_reference()
        self.detector.start()
        self.stream_server.start()

        prev_time = time.monotonic()
        fps = 0.0
        is_pi = self.config.is_pi()

        try:
            while True:
                altitude_delta = 0.0
                if self.barometer:
                    altitude_delta = self.barometer.get_altitude_delta_m()
                is_airborne = self.detect_only or altitude_delta > self.config.altitude_margin_m

                _, detections = self.detector.get_frame_and_detections()

                # Push video stream
                if is_pi:
                    jpeg = self.detector.get_jpeg()
                    if jpeg:
                        self.stream_server.push_jpeg(jpeg)
                else:
                    frame, _ = self.detector.get_frame_and_detections()
                    if frame is not None:
                        self.stream_server.push_frame(frame)

                if detections is None:
                    time.sleep(0.001)
                    continue

                if not is_airborne:
                    detections = []

                # Servo rearm check
                if self.servo and self.servo.is_ready and self.activation._cooling_down:
                    self.activation.mark_ready()
                    print("*** REARMED — ready to fire again ***")

                # Activation filter
                img = self.config.img_size
                state = self.activation.update(altitude_delta, detections, img, img)

                # Fire
                if not self.detect_only and state.armed and self.servo and self.servo.is_ready:
                    self.servo.fire()
                    self.activation.mark_fired()
                    print("*** NET DEPLOYED ***")

                # FPS
                now = time.monotonic()
                dt = now - prev_time
                if dt > 0:
                    fps = 1.0 / dt
                prev_time = now

                self.stream_server.push_telemetry(
                    altitude_delta, detections, state, fps
                )

        except KeyboardInterrupt:
            print("\nShutdown requested")
        finally:
            self.detector.stop()
            if self.servo:
                self.servo.safe()
            self.stream_server.stop()
            print("=== Drone Guardian stopped ===")
