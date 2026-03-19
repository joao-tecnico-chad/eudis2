"""Abstract base for servo actuators."""

from abc import ABC, abstractmethod


class ServoABC(ABC):
    @abstractmethod
    def arm(self) -> None:
        """Move servo to armed position (ready to fire)."""

    @abstractmethod
    def fire(self) -> None:
        """Release the net launcher."""

    @abstractmethod
    def safe(self) -> None:
        """Move servo to safe/stowed position."""
