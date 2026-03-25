"""Abstract base for servo actuators."""

from abc import ABC, abstractmethod


class ServoABC(ABC):
    @abstractmethod
    def fire(self) -> None:
        """Full fire cycle: rotate 90 deg, wait 1s, rotate back 90 deg (rearm)."""

    @abstractmethod
    def safe(self) -> None:
        """Move servo to safe/stowed position."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if servo has completed rearm and is ready to fire again."""
