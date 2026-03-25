"""Desktop stub servo — logs actions without hardware."""

import threading
import time

from guardian.actuators.base import ServoABC


class StubServo(ServoABC):
    """Simulates the fire-rearm cycle for desktop testing."""

    REARM_DELAY_S = 1.0

    def __init__(self):
        self._ready = True
        self._cycle_thread: threading.Thread | None = None

    def fire(self) -> None:
        if not self._ready:
            print("[STUB SERVO] Still in fire cycle, ignoring")
            return

        self._ready = False
        print("[STUB SERVO] FIRED — net deployed!")

        self._cycle_thread = threading.Thread(target=self._rearm_cycle, daemon=True)
        self._cycle_thread.start()

    def _rearm_cycle(self) -> None:
        time.sleep(self.REARM_DELAY_S)
        self._ready = True
        print("[STUB SERVO] REARMED — ready to fire again")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def safe(self) -> None:
        if self._cycle_thread is not None:
            self._cycle_thread.join(timeout=3.0)
        self._ready = True
        print("[STUB SERVO] SAFE")
