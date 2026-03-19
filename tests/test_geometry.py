"""Tests for geometry utilities."""

import math

from guardian.utils.geometry import (
    box_area_ratio,
    box_centroid,
    estimate_distance,
    estimate_focal_length,
    is_in_center_zone,
)


def test_focal_length_90_fov():
    # 90 degree FOV: focal_length = width / (2 * tan(45)) = width / 2
    fl = estimate_focal_length(640, 90.0)
    assert abs(fl - 320.0) < 0.01


def test_focal_length_120_fov():
    fl = estimate_focal_length(640, 120.0)
    expected = 640 / (2 * math.tan(math.radians(60)))
    assert abs(fl - expected) < 0.01


def test_estimate_distance():
    # Object 100px wide at focal length 500, real width 0.35m
    dist = estimate_distance(500.0, 0.35, 100)
    assert abs(dist - 1.75) < 0.01


def test_estimate_distance_zero_width():
    assert estimate_distance(500.0, 0.35, 0) == -1.0


def test_center_zone_center():
    # Dead center should always be in the zone
    assert is_in_center_zone(320, 240, 640, 480, 0.30)


def test_center_zone_corner():
    # Top-left corner should NOT be in central 30%
    assert not is_in_center_zone(10, 10, 640, 480, 0.30)


def test_center_zone_boundary():
    # On the exact boundary (35% of 640 = 224)
    assert is_in_center_zone(224, 240, 640, 480, 0.30)
    assert not is_in_center_zone(223, 240, 640, 480, 0.30)


def test_box_area_ratio():
    # 100x100 box in 1000x1000 frame = 1%
    ratio = box_area_ratio(0, 0, 100, 100, 1000, 1000)
    assert abs(ratio - 0.01) < 0.0001


def test_box_area_ratio_full_frame():
    ratio = box_area_ratio(0, 0, 640, 480, 640, 480)
    assert abs(ratio - 1.0) < 0.0001


def test_box_centroid():
    cx, cy = box_centroid(100, 200, 300, 400)
    assert cx == 200.0
    assert cy == 300.0
