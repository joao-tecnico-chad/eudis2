"""PWM servo control on Raspberry Pi GPIO."""

import threading
import time

from guardian.actuators.base import ServoABC
from guardian.config import GuardianConfig


class PWMServo(ServoABC):
    """Controls the net launcher servo via GPIO PWM on the RPi Zero 2W.

    Fire cycle (non-blocking):
        1. Rotate 90 deg (fire position)
        2. Wait 1 second
        3. Rotate 90 deg back (rearm position)
        4. Ready to fire again
    """

    REARM_DELAY_S = 1.0

    def __init__(self, config: GuardianConfig):
        self._gpio = config.servo_gpio
        self._rest_angle = config.servo_arm_angle      # 0 deg — resting / rearmed
        self._fire_angle = config.servo_fire_angle      # 90 deg — fired
        self._servo = None
        self._ready = True
        self._cycle_thread: threading.Thread | None = None

    def _ensure_init(self) -> None:
        if self._servo is not None:
            return
        from gpiozero import Servo as GpioServo
        from gpiozero.pins.pigpio import PiGPIOFactory

        try:
            factory = PiGPIOFactory()
            self._servo = GpioServo(self._gpio, pin_factory=factory)
        except Exception:
            self._servo = GpioServo(self._gpio)
        print(f"PWM servo initialized on GPIO{self._gpio}")

    def _set_angle(self, angle: float) -> None:
        """Map 0-180 deg to gpiozero's -1..+1 range."""
        self._ensure_init()
        value = (angle / 90.0) - 1.0
        value = max(-1.0, min(1.0, value))
        self._servo.value = value

    def fire(self) -> None:
        if not self._ready:
            print("Servo: still in fire cycle, ignoring")
            return

        self._ready = False
        self._set_angle(self._fire_angle)
        print("Servo: FIRED")

        # Run rearm in background so main loop isn't blocked
        self._cycle_thread = threading.Thread(target=self._rearm_cycle, daemon=True)
        self._cycle_thread.start()

    def _rearm_cycle(self) -> None:
        time.sleep(self.REARM_DELAY_S)
        self._set_angle(self._rest_angle)
        self._ready = True
        print("Servo: REARMED — ready to fire again")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def safe(self) -> None:
        # Wait for any in-progress cycle to finish
        if self._cycle_thread is not None:
            self._cycle_thread.join(timeout=3.0)
        self._set_angle(self._rest_angle)
        self._ready = True
        if self._servo is not None:
            self._servo.detach()
        print("Servo: SAFE")
