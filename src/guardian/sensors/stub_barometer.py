"""Desktop stub barometer — returns configurable altitude for testing."""

import time

from guardian.sensors.base import BarometerABC


class StubBarometer(BarometerABC):
    """Simulates altitude for desktop testing.

    By default, simulates a drone climbing to 50m over 10 seconds then holding.
    Set simulate_flight=False for a fixed altitude.
    """

    def __init__(self, fixed_altitude: float = 50.0, simulate_flight: bool = True):
        self._fixed_altitude = fixed_altitude
        self._simulate_flight = simulate_flight
        self._reference_alt = 0.0
        self._start_time = time.monotonic()

    def set_reference(self) -> None:
        self._reference_alt = 0.0
        self._start_time = time.monotonic()
        print(f"Stub barometer: reference set to {self._reference_alt:.1f}m")

    def read_altitude_m(self) -> float:
        if not self._simulate_flight:
            return self._fixed_altitude

        # Simulate climb: 0 -> fixed_altitude over 10 seconds, then hold
        elapsed = time.monotonic() - self._start_time
        climb_duration = 10.0
        if elapsed >= climb_duration:
            return self._fixed_altitude
        return self._fixed_altitude * (elapsed / climb_duration)

    def get_altitude_delta_m(self) -> float:
        return self.read_altitude_m() - self._reference_alt
