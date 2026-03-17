"""
Convert trained YOLOv6n weights to .blob for OAK-1W deployment

Steps:
    .pt → ONNX → OpenVINO IR → .blob (Myriad X)

Requirements:
    pip install blobconverter openvino-dev onnx onnxsim

Usage:
    python convert_to_blob.py
    python convert_to_blob.py --weights path/to/best.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_WEIGHTS = Path("YOLOv6/runs/train/drone_yolov6n/weights/best.pt")
ONNX_PATH = Path("drone_yolov6n.onnx")
BLOB_PATH = Path("drone_yolov6n.blob")
IMG_SIZE = 416
SHAVES = 6


def export_to_onnx(weights: Path) -> None:
    print(f"Exporting {weights} to ONNX...")
    subprocess.run(
        [
            sys.executable, "deploy/ONNX/export_onnx.py",
            "--weights", str(weights.resolve()),
            "--img-size", str(IMG_SIZE), str(IMG_SIZE),
            "--batch-size", "1",
            "--simplify",
            "--save-dir", str(ONNX_PATH.parent.resolve()),
        ],
        cwd="YOLOv6",
        check=True,
    )
    # YOLOv6 export saves as best.onnx — rename to our target
    exported = weights.parent / "best.onnx"
    if exported.exists():
        exported.rename(ONNX_PATH)
    print(f"ONNX saved to {ONNX_PATH}")


def convert_to_blob() -> None:
    print(f"Converting ONNX to .blob for OAK-1W ({SHAVES} shaves, FP16)...")
    import blobconverter

    blob_path = blobconverter.from_onnx(
        model=str(ONNX_PATH),
        data_type="FP16",
        shaves=SHAVES,
        version="2022.1",
        output_dir=".",
        optimizer_params=[
            f"--input_shape=[1,3,{IMG_SIZE},{IMG_SIZE}]",
            "--data_type=FP16",
        ],
    )

    Path(blob_path).rename(BLOB_PATH)
    print(f"\nBlob saved to {BLOB_PATH}")
    print("Ready to deploy on OAK-1W — run detect_drone_oak.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLOv6n to OAK-1W blob")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt")
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"ERROR: Weights not found at {args.weights}")
        print("Run train.py first, or pass --weights path/to/best.pt")
        sys.exit(1)

    if not ONNX_PATH.exists():
        export_to_onnx(args.weights)

    convert_to_blob()


if __name__ == "__main__":
    main()
