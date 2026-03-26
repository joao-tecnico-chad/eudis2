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
    def __init__(self, config: GuardianConfig):
        self.config = config
        self.detector = self._create_detector()
        self.barometer = self._create_barometer()
        self.servo = self._create_servo()
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
        is_pi = self.config.is_pi()

        try:
            while True:
                altitude_delta = self.barometer.get_altitude_delta_m()
                is_airborne = altitude_delta > self.config.altitude_margin_m

                _, detections = self.detector.get_frame_and_detections()

                # On Pi, push hardware-encoded MJPEG (zero CPU cost)
                # On desktop, push_frame does CPU encoding (fallback)
                if is_pi:
                    jpeg = self.detector.get_jpeg()
                    if jpeg:
                        self.stream_server.push_jpeg(jpeg)
                else:
                    # Desktop stub returns frames, need CPU encode
                    frame, _ = self.detector.get_frame_and_detections()
                    if frame is not None:
                        self.stream_server.push_frame(frame)

                if detections is None:
                    time.sleep(0.001)
                    continue

                if not is_airborne:
                    detections = []

                # Servo rearm check
                if self.servo.is_ready and self.activation._cooling_down:
                    self.activation.mark_ready()
                    print("*** REARMED — ready to fire again ***")

                # Activation filter
                img = self.config.img_size
                state = self.activation.update(altitude_delta, detections, img, img)

                # Fire
                if state.armed and self.servo.is_ready:
                    self.servo.fire()
                    self.activation.mark_fired()
                    print("*** NET DEPLOYED ***")

                # FPS
                now = time.monotonic()
                dt = now - prev_time
                if dt > 0:
                    fps = 1.0 / dt
                prev_time = now

                # Push telemetry (detections as JSON for browser-side box rendering)
                self.stream_server.push_telemetry(
                    altitude_delta, detections, state, fps
                )

        except KeyboardInterrupt:
            print("\nShutdown requested")
        finally:
            self.detector.stop()
            self.servo.safe()
            self.stream_server.stop()
            print("=== Drone Guardian stopped ===")
