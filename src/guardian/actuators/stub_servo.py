"""Desktop stub servo — logs actions without hardware."""

from guardian.actuators.base import ServoABC


class StubServo(ServoABC):
    def __init__(self):
        self._state = "safe"

    def arm(self) -> None:
        if self._state == "fired":
            print("[STUB SERVO] Already fired, cannot re-arm")
            return
        self._state = "armed"
        print("[STUB SERVO] ARMED")

    def fire(self) -> None:
        if self._state != "armed":
            print(f"[STUB SERVO] Cannot fire from state '{self._state}'")
            return
        self._state = "fired"
        print("[STUB SERVO] FIRED -- net deployed!")

    def safe(self) -> None:
        self._state = "safe"
        print("[STUB SERVO] SAFE")
