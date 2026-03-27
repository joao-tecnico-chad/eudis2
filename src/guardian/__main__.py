"""Entry point: python -m guardian"""

import argparse

from guardian.config import GuardianConfig
from guardian.main import DroneGuardian


def main():
    parser = argparse.ArgumentParser(description="Drone Guardian")
    parser.add_argument("--hardware-mode", choices=["pi", "desktop", "auto"], default="auto")
    parser.add_argument("--model", default="models/emine_yolov6s.rvc2.tar.xz",
                        help="Path to NNArchive (.tar.xz) or .onnx model")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--detect-only", action="store_true",
                        help="Detection + streaming only, no barometer/servo")
    args = parser.parse_args()

    config = GuardianConfig(
        hardware_mode=args.hardware_mode,
        model_path=args.model,
        conf_threshold=args.conf,
        stream_port=args.port,
    )

    guardian = DroneGuardian(config, detect_only=args.detect_only)
    guardian.run()


if __name__ == "__main__":
    main()
