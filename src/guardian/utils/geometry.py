"""Geometry utilities for drone detection and activation logic."""

import math


def estimate_focal_length(frame_width: int, fov_degrees: float) -> float:
    """Compute focal length in pixels from frame width and horizontal FOV."""
    return frame_width / (2 * math.tan(math.radians(fov_degrees / 2)))


def estimate_distance(focal_length: float, real_width_m: float, pixel_width: int) -> float:
    """Estimate distance to an object using the pinhole camera model.

    Returns distance in meters, or -1 if pixel_width is invalid.
    """
    if pixel_width <= 0:
        return -1.0
    return (real_width_m * focal_length) / pixel_width


def is_in_center_zone(cx: float, cy: float, frame_w: int, frame_h: int,
                      zone_ratio: float) -> bool:
    """Check if a point (cx, cy) is within the central zone of the frame.

    zone_ratio=0.30 means the central 30% band in each dimension.
    """
    margin_x = (1.0 - zone_ratio) / 2.0
    margin_y = (1.0 - zone_ratio) / 2.0
    x_min = margin_x * frame_w
    x_max = (1.0 - margin_x) * frame_w
    y_min = margin_y * frame_h
    y_max = (1.0 - margin_y) * frame_h
    return x_min <= cx <= x_max and y_min <= cy <= y_max


def box_area_ratio(x1: int, y1: int, x2: int, y2: int, frame_w: int, frame_h: int) -> float:
    """Compute the ratio of bounding box area to total frame area."""
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = frame_w * frame_h
    if frame_area == 0:
        return 0.0
    return box_area / frame_area


def box_centroid(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    """Compute the centroid of a bounding box."""
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
