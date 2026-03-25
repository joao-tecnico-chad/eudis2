"""4-layer activation filter for net launcher safety.

Layer 1 - Altitude:    Only active when airborne (altitude > reference + margin)
Layer 2 - Zone:        Drone centroid must be in central zone of frame
Layer 3 - Size:        Bounding box area must exceed minimum ratio (drone is close)
Layer 4 - Persistence: N consecutive frames with all layers passing
"""

from dataclasses import dataclass

from guardian.config import GuardianConfig
from guardian.utils.decode import Detection
from guardian.utils.geometry import box_area_ratio, box_centroid, is_in_center_zone


@dataclass
class ActivationState:
    layer1_altitude: bool = False
    layer2_centroid: bool = False
    layer3_size: bool = False
    layer4_count: int = 0
    armed: bool = False
    cooling_down: bool = False

    @property
    def all_layers_passing(self) -> bool:
        return self.layer1_altitude and self.layer2_centroid and self.layer3_size


class ActivationFilter:
    """Stateful 4-layer activation filter.

    Call update() once per frame with current sensor data and detections.
    The filter tracks consecutive passing frames and signals when to fire.
    After firing, enter cooldown until the servo signals ready (mark_ready).
    """

    def __init__(self, config: GuardianConfig):
        self._altitude_margin = config.altitude_margin_m
        self._zone_ratio = config.centroid_zone_ratio
        self._min_area_ratio = config.min_box_area_ratio
        self._required_frames = config.consecutive_frames
        self._consecutive = 0
        self._cooling_down = False

    def reset(self) -> None:
        """Reset filter state."""
        self._consecutive = 0
        self._cooling_down = False

    def update(self, altitude_delta_m: float, detections: list[Detection],
               frame_w: int, frame_h: int) -> ActivationState:
        state = ActivationState()

        # Layer 1: Altitude gate (master safety)
        state.layer1_altitude = altitude_delta_m > self._altitude_margin
        if not state.layer1_altitude:
            self._consecutive = 0
            state.cooling_down = self._cooling_down
            return state

        # If servo is still cycling, don't accumulate frames
        if self._cooling_down:
            self._consecutive = 0
            state.cooling_down = True
            return state

        # Layer 2 & 3: Check each detection for zone and size
        for det in detections:
            cx, cy = box_centroid(det.x1, det.y1, det.x2, det.y2)
            in_zone = is_in_center_zone(cx, cy, frame_w, frame_h, self._zone_ratio)
            area = box_area_ratio(det.x1, det.y1, det.x2, det.y2, frame_w, frame_h)

            if in_zone:
                state.layer2_centroid = True
            if area > self._min_area_ratio:
                state.layer3_size = True

            if in_zone and area > self._min_area_ratio:
                break

        # Layer 4: Persistence
        if state.layer2_centroid and state.layer3_size:
            self._consecutive += 1
        else:
            self._consecutive = 0

        state.layer4_count = self._consecutive

        if self._consecutive >= self._required_frames:
            state.armed = True

        return state

    def mark_fired(self) -> None:
        """Enter cooldown while servo completes fire-rearm cycle."""
        self._cooling_down = True
        self._consecutive = 0

    def mark_ready(self) -> None:
        """Servo has rearmed — allow activation again."""
        self._cooling_down = False
        self._consecutive = 0
