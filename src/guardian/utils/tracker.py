"""Single-target detection tracker with temporal hold and confidence smoothing."""

from dataclasses import dataclass

from guardian.utils.decode import Detection


@dataclass
class TrackState:
    """Current tracking state returned by DetectionTracker.update()."""
    tracking: bool = False
    confirmed: bool = False
    hold_time: float = 0.0
    ema_conf: float = 0.0
    peak_conf: float = 0.0
    best_det: Detection | None = None
    cx: float = 0.0
    cy: float = 0.0
    lost_reason: str = ""


class DetectionTracker:
    """Track a single detection target across frames.

    Pipeline per frame:
        1. Filter oversized boxes (full-frame false positives)
        2. Spatial match (nearest to tracked centroid, or highest confidence if new)
        3. EMA confidence smoothing
        4. Confidence drop detection (track lost if EMA drops below 40% of peak)
        5. Gap tolerance (no match for gap_sec resets track)
        6. Temporal hold (confirmed after hold_sec continuous tracking)
    """

    def __init__(
        self,
        img_size: int = 640,
        hold_sec: float = 3.0,
        gap_sec: float = 0.5,
        match_frac: float = 0.25,
        max_box_ratio: float = 0.6,
        ema_alpha: float = 0.3,
        conf_drop: float = 0.4,
    ):
        self._img_size = img_size
        self._hold_sec = hold_sec
        self._gap_sec = gap_sec
        self._match_dist = img_size * match_frac
        self._max_box_ratio = max_box_ratio
        self._ema_alpha = ema_alpha
        self._conf_drop = conf_drop

        # Internal state
        self._track_cx = 0.0
        self._track_cy = 0.0
        self._detect_start: float | None = None
        self._last_detect: float | None = None
        self._confirmed = False
        self._ema_conf = 0.0
        self._peak_conf = 0.0

    def reset(self) -> None:
        self._detect_start = None
        self._last_detect = None
        self._confirmed = False
        self._ema_conf = 0.0
        self._peak_conf = 0.0

    def update(self, detections: list[Detection], now: float) -> TrackState:
        state = TrackState()

        # Step 1: Filter oversized boxes
        filtered = [
            d for d in detections
            if (d.x2 - d.x1) / self._img_size <= self._max_box_ratio
            and (d.y2 - d.y1) / self._img_size <= self._max_box_ratio
        ]

        # Step 2: Spatial match
        best = self._match(filtered)

        if best is not None:
            cx = (best.x1 + best.x2) / 2.0
            cy = (best.y1 + best.y2) / 2.0

            # Step 3: EMA confidence
            if self._detect_start is None:
                self._ema_conf = best.confidence
            else:
                self._ema_conf = (
                    self._ema_alpha * best.confidence
                    + (1 - self._ema_alpha) * self._ema_conf
                )
            self._peak_conf = max(self._peak_conf, self._ema_conf)

            # Step 4: Confidence drop detection
            if (
                self._confirmed
                and self._peak_conf > 0
                and self._ema_conf < self._peak_conf * self._conf_drop
            ):
                state.lost_reason = (
                    f"conf {self._peak_conf:.0%} -> {self._ema_conf:.0%}"
                )
                self.reset()
                return state

            # Update track
            self._track_cx = cx
            self._track_cy = cy
            if self._detect_start is None:
                self._detect_start = now
            self._last_detect = now

            hold_time = now - self._detect_start
            if hold_time >= self._hold_sec:
                self._confirmed = True

            state.tracking = True
            state.confirmed = self._confirmed
            state.hold_time = hold_time
            state.ema_conf = self._ema_conf
            state.peak_conf = self._peak_conf
            state.best_det = best
            state.cx = cx
            state.cy = cy

        # Step 5: Gap tolerance
        if self._last_detect is not None and (now - self._last_detect) > self._gap_sec:
            if self._confirmed:
                state.lost_reason = "gap timeout"
            self.reset()
            state.tracking = False
            state.confirmed = False

        return state

    def _match(self, detections: list[Detection]) -> Detection | None:
        if not detections:
            return None

        if self._detect_start is not None:
            # Active track — find nearest to tracked centroid
            def dist(d: Detection) -> float:
                cx = (d.x1 + d.x2) / 2.0
                cy = (d.y1 + d.y2) / 2.0
                return ((cx - self._track_cx) ** 2 + (cy - self._track_cy) ** 2) ** 0.5

            nearest = min(detections, key=dist)
            if dist(nearest) < self._match_dist:
                return nearest
            return None
        else:
            # No active track — pick highest confidence
            return max(detections, key=lambda d: d.confidence)
