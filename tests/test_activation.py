"""Tests for the 4-layer activation filter."""

from guardian.activation.filter import ActivationFilter, ActivationState
from guardian.config import GuardianConfig
from guardian.utils.decode import Detection


def make_config(**overrides) -> GuardianConfig:
    defaults = dict(
        altitude_margin_m=5.0,
        centroid_zone_ratio=0.30,
        min_box_area_ratio=0.01,
        consecutive_frames=5,
    )
    defaults.update(overrides)
    return GuardianConfig(**defaults)


def make_detection(x1, y1, x2, y2, conf=0.9, cls=0) -> Detection:
    return Detection(x1, y1, x2, y2, conf, cls)


# Frame dimensions for tests
FW, FH = 640, 480


class TestLayer1Altitude:
    def test_below_threshold_resets_all(self):
        filt = ActivationFilter(make_config())
        det = make_detection(280, 200, 360, 280)  # center, >1% area
        # Build up some consecutive frames
        for _ in range(3):
            filt.update(10.0, [det], FW, FH)
        # Drop altitude below threshold
        state = filt.update(3.0, [det], FW, FH)
        assert not state.layer1_altitude
        assert state.layer4_count == 0

    def test_above_threshold_passes(self):
        filt = ActivationFilter(make_config())
        det = make_detection(280, 200, 360, 280)
        state = filt.update(10.0, [det], FW, FH)
        assert state.layer1_altitude


class TestLayer2Zone:
    def test_center_detection_passes(self):
        filt = ActivationFilter(make_config())
        # Detection centered in frame
        det = make_detection(280, 200, 360, 280)
        state = filt.update(10.0, [det], FW, FH)
        assert state.layer2_centroid

    def test_corner_detection_fails(self):
        filt = ActivationFilter(make_config())
        # Detection in top-left corner
        det = make_detection(10, 10, 50, 50)
        state = filt.update(10.0, [det], FW, FH)
        assert not state.layer2_centroid


class TestLayer3Size:
    def test_large_detection_passes(self):
        filt = ActivationFilter(make_config())
        # 80x80 = 6400 pixels, frame = 640*480 = 307200, ratio = 2.1% > 1%
        det = make_detection(280, 200, 360, 280)
        state = filt.update(10.0, [det], FW, FH)
        assert state.layer3_size

    def test_tiny_detection_fails(self):
        filt = ActivationFilter(make_config())
        # 5x5 = 25 pixels, ratio = 0.008% < 1%
        det = make_detection(320, 240, 325, 245)
        state = filt.update(10.0, [det], FW, FH)
        assert not state.layer3_size


class TestLayer4Persistence:
    def test_fires_after_n_frames(self):
        config = make_config(consecutive_frames=5)
        filt = ActivationFilter(config)
        det = make_detection(280, 200, 360, 280)  # center, large enough

        for i in range(4):
            state = filt.update(10.0, [det], FW, FH)
            assert not state.armed, f"Should not be armed at frame {i+1}"

        state = filt.update(10.0, [det], FW, FH)
        assert state.armed
        assert state.layer4_count == 5

    def test_resets_on_missed_frame(self):
        config = make_config(consecutive_frames=5)
        filt = ActivationFilter(config)
        det_center = make_detection(280, 200, 360, 280)
        det_corner = make_detection(10, 10, 50, 50)

        for _ in range(3):
            filt.update(10.0, [det_center], FW, FH)

        # One frame with no qualifying detection resets counter
        filt.update(10.0, [det_corner], FW, FH)

        state = filt.update(10.0, [det_center], FW, FH)
        assert state.layer4_count == 1  # Reset, started fresh

    def test_no_detections_resets(self):
        config = make_config(consecutive_frames=5)
        filt = ActivationFilter(config)
        det = make_detection(280, 200, 360, 280)

        for _ in range(3):
            filt.update(10.0, [det], FW, FH)

        state = filt.update(10.0, [], FW, FH)
        assert state.layer4_count == 0


class TestFiring:
    def test_mark_fired_prevents_rearm(self):
        config = make_config(consecutive_frames=2)
        filt = ActivationFilter(config)
        det = make_detection(280, 200, 360, 280)

        filt.update(10.0, [det], FW, FH)
        state = filt.update(10.0, [det], FW, FH)
        assert state.armed

        filt.mark_fired()
        state = filt.update(10.0, [det], FW, FH)
        assert not state.armed
        assert state.fired

    def test_reset_allows_refire(self):
        config = make_config(consecutive_frames=2)
        filt = ActivationFilter(config)
        det = make_detection(280, 200, 360, 280)

        filt.update(10.0, [det], FW, FH)
        filt.update(10.0, [det], FW, FH)
        filt.mark_fired()
        filt.reset()

        filt.update(10.0, [det], FW, FH)
        state = filt.update(10.0, [det], FW, FH)
        assert state.armed
        assert not state.fired
