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
        self.barometer = None if detect_only else self._create_barometer()
        self.servo = None if detect_only else self._create_servo()
        self.activation = ActivationFilter(config)
        self.stream_server = StreamServer(config)

    def _create_detector(self):
        if self.config.is_pi():
            from guardian.detection.oak_detector import OakDetector
            return OakDetector(self.config)
        else:
            from guardian.detection.stub_detector import StubDetector
            return StubDetector(self.config)

    def _create_barometer(self):
        if self.config.is_pi():
            from guardian.sensors.bmp390 import BMP390Barometer
            return BMP390Barometer()
        else:
            from guardian.sensors.stub_barometer import StubBarometer
            return StubBarometer()

    def _create_servo(self):
        if self.config.is_pi():
            from guardian.actuators.pwm_servo import PWMServo
            return PWMServo(self.config)
        else:
            from guardian.actuators.stub_servo import StubServo
            return StubServo()

    def run(self) -> None:
        print("=== Drone Guardian ===")
        mode = "detect-only" if self.detect_only else ("Pi" if self.config.is_pi() else "Desktop")
        print(f"Mode: {mode}")

        if self.barometer:
            self.barometer.set_reference()
        self.detector.start()
        self.stream_server.start()

        prev_time = time.monotonic()
        fps = 0.0
        frame_count = 0
        fps_time = time.monotonic()
        is_pi = self.config.is_pi()

        try:
            while True:
                # Stream video
                if is_pi:
                    jpeg = self.detector.get_jpeg()
                    if jpeg:
                        self.stream_server.push_jpeg(jpeg)

                # Get detections
                _, detections = self.detector.get_frame_and_detections()
                if detections is None:
                    # Desktop: also stream frames
                    if not is_pi:
                        frame, _ = self.detector.get_frame_and_detections()
                        if frame is not None:
                            self.stream_server.push_frame(frame)
                    time.sleep(0.001)
                    continue

                # FPS
                frame_count += 1
                now = time.monotonic()
                if now - fps_time >= 2.0:
                    fps = frame_count / (now - fps_time)
                    frame_count = 0
                    fps_time = now

                # Altitude gate
                altitude_delta = 0.0
                if self.barometer:
                    altitude_delta = self.barometer.get_altitude_delta_m()
                is_airborne = self.detect_only or altitude_delta > self.config.altitude_margin_m

                if not is_airborne:
                    detections = []

                if detections:
                    best = max(detections, key=lambda d: d.confidence)
                    print(f"\r  DRONE {best.confidence:.0%} ({len(detections)} det) [{fps:.0f} fps]   ", end="", flush=True)

                # Activation filter
                img = self.config.img_size
                state = self.activation.update(altitude_delta, detections, img, img)

                # Servo control
                if not self.detect_only:
                    if self.servo and self.servo.is_ready and self.activation._cooling_down:
                        self.activation.mark_ready()
                        print("\n*** REARMED ***")
                    if state.armed and self.servo and self.servo.is_ready:
                        self.servo.fire()
                        self.activation.mark_fired()
                        print("\n*** NET DEPLOYED ***")

                # Push telemetry to dashboard
                self.stream_server.push_telemetry(
                    altitude_delta, detections, state, fps
                )

        except KeyboardInterrupt:
            print("\nShutdown")
        finally:
            self.detector.stop()
            if self.servo:
                self.servo.safe()
            self.stream_server.stop()
