"""PWM servo control on Raspberry Pi GPIO."""

import time

from guardian.actuators.base import ServoABC
from guardian.config import GuardianConfig


class PWMServo(ServoABC):
    """Controls the net launcher servo via GPIO PWM on the RPi Zero 2W.

    Uses gpiozero for hardware PWM on GPIO13 (physical pin 33).
    CW rotation arms the door, CCW fires (releases the latch).
    """

    def __init__(self, config: GuardianConfig):
        self._gpio = config.servo_gpio
        self._arm_angle = config.servo_arm_angle
        self._fire_angle = config.servo_fire_angle
        self._servo = None
        self._state = "safe"

    def _ensure_init(self) -> None:
        if self._servo is not None:
            return
        from gpiozero import Servo as GpioServo
        from gpiozero.pins.pigpio import PiGPIOFactory

        # pigpio provides hardware-timed PWM (less jitter than software PWM)
        try:
            factory = PiGPIOFactory()
            self._servo = GpioServo(self._gpio, pin_factory=factory)
        except Exception:
            # Fall back to default pin factory
            self._servo = GpioServo(self._gpio)
        print(f"PWM servo initialized on GPIO{self._gpio}")

    def _set_angle(self, angle: float) -> None:
        """Set servo position. Maps 0-180 degrees to gpiozero's -1 to +1 range."""
        self._ensure_init()
        value = (angle / 90.0) - 1.0  # 0deg=-1, 90deg=0, 180deg=1
        value = max(-1.0, min(1.0, value))
        self._servo.value = value

    def arm(self) -> None:
        if self._state == "fired":
            print("Servo: already fired, cannot re-arm")
            return
        self._set_angle(self._arm_angle)
        self._state = "armed"
        print("Servo: ARMED")

    def fire(self) -> None:
        if self._state != "armed":
            print(f"Servo: cannot fire from state '{self._state}'")
            return
        self._set_angle(self._fire_angle)
        self._state = "fired"
        print("Servo: FIRED")
        time.sleep(0.5)  # Allow servo to complete movement

    def safe(self) -> None:
        self._set_angle(self._arm_angle)
        self._state = "safe"
        if self._servo is not None:
            self._servo.detach()
        print("Servo: SAFE")
