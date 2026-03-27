"""Entry point for: python -m guardian"""

import argparse

from guardian.config import GuardianConfig
from guardian.main import DroneGuardian


def main():
    parser = argparse.ArgumentParser(
        description="Drone Guardian — Autonomous Interceptor Drone Detection & Neutralisation"
    )
    parser.add_argument("--hardware-mode", choices=["pi", "desktop", "auto"], default="auto",
                        help="Hardware mode (default: auto-detect)")
    parser.add_argument("--source", default="0",
                        help="Video source for desktop mode: webcam index or file path")
    parser.add_argument("--blob", default="models/best_yolov6n_openvino_2022.1_6shave.blob",
                        help="Path to .blob model for OAK-1W")
    parser.add_argument("--onnx", default="models/drone_yolov6n.onnx",
                        help="Path to .onnx model for desktop testing")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Detection confidence threshold")
    parser.add_argument("--port", type=int, default=8080,
                        help="Streaming server port")
    parser.add_argument("--jpeg-quality", type=int, default=50,
                        help="JPEG quality for stream (1-100)")
    parser.add_argument("--model-format", choices=["yolov6", "yolov8"], default="yolov6",
                        help="Model architecture format")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Model input size")
    parser.add_argument("--detect-only", action="store_true",
                        help="Detection + streaming only, no barometer/servo")
    args = parser.parse_args()

    config = GuardianConfig(
        hardware_mode=args.hardware_mode,
        source=args.source,
        blob_path=args.blob,
        onnx_path=args.onnx,
        conf_threshold=args.conf,
        stream_port=args.port,
        jpeg_quality=args.jpeg_quality,
        model_format=args.model_format,
        img_size=args.img_size,
    )

    guardian = DroneGuardian(config, detect_only=args.detect_only)
    guardian.run()


if __name__ == "__main__":
    main()
