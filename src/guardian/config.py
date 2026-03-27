"""Guardian configuration."""

import platform
from dataclasses import dataclass, field


@dataclass
class GuardianConfig:
    # --- Detection ---
    model_path: str = "models/emine_yolov6s.rvc2.tar.xz"
    img_size: int = 640
    conf_threshold: float = 0.3
    iou_threshold: float = 0.5
    max_box_ratio: float = 0.6
    num_classes: int = 1
    labels: list[str] = field(default_factory=lambda: ["drone"])

    # --- Camera ---
    camera_fov_deg: float = 120.0
    real_drone_width_m: float = 0.35

    # --- Activation ---
    altitude_margin_m: float = 5.0
    centroid_zone_ratio: float = 0.30
    min_box_area_ratio: float = 0.01
    consecutive_frames: int = 5

    # --- Servo ---
    servo_gpio: int = 18
    servo_arm_angle: float = 90.0
    servo_fire_angle: float = 135.0

    # --- Streaming ---
    stream_host: str = "0.0.0.0"
    stream_port: int = 8080
    jpeg_quality: int = 70

    # --- Hardware ---
    hardware_mode: str = "auto"  # "pi", "desktop", "auto"
    source: str = "0"

    def is_pi(self) -> bool:
        if self.hardware_mode == "pi":
            return True
        if self.hardware_mode == "desktop":
            return False
        return platform.machine().startswith("aarch64") or platform.machine().startswith("arm")
