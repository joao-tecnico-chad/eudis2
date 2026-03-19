"""Abstract base for barometer sensors."""

from abc import ABC, abstractmethod


class BarometerABC(ABC):
    @abstractmethod
    def set_reference(self) -> None:
        """Record current altitude as ground-level reference."""

    @abstractmethod
    def read_altitude_m(self) -> float:
        """Read current altitude in meters above sea level."""

    @abstractmethod
    def get_altitude_delta_m(self) -> float:
        """Get altitude relative to reference (current - reference)."""
