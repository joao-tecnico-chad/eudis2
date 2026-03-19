"""Central configuration for Drone Guardian."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GuardianConfig:
    # --- Detection ---
    blob_path: str = "models/drone_yolov8n.blob"
    onnx_path: str = "models/best_new.onnx"
    img_size: int = 416
    conf_threshold: float = 0.3
    iou_threshold: float = 0.5
    max_box_ratio: float = 0.6
    model_format: str = "yolov6"  # "yolov6" or "yolov8"
    num_classes: int = 1
    labels: list[str] = field(default_factory=lambda: ["drone"])

    # --- Camera ---
    camera_fov_deg: float = 120.0  # OAK-1W IMX378
    real_drone_width_m: float = 0.35

    # --- Activation layers ---
    altitude_margin_m: float = 5.0
    centroid_zone_ratio: float = 0.30
    min_box_area_ratio: float = 0.01
    consecutive_frames: int = 5

    # --- Servo ---
    servo_gpio: int = 13
    servo_arm_angle: float = 0.0
    servo_fire_angle: float = 90.0

    # --- Streaming ---
    stream_host: str = "0.0.0.0"
    stream_port: int = 8080
    jpeg_quality: int = 50

    # --- Hardware mode ---
    hardware_mode: str = "auto"  # "pi", "desktop", "auto"
    source: str = "0"  # video source for desktop mode (webcam index or file)

    def is_pi(self) -> bool:
        if self.hardware_mode == "pi":
            return True
        if self.hardware_mode == "desktop":
            return False
        # Auto-detect
        try:
            with open("/proc/device-tree/model") as f:
                return "raspberry pi" in f.read().lower()
        except (FileNotFoundError, PermissionError):
            return False
