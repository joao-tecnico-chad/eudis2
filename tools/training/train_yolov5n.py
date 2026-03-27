"""
Train YOLOv5n drone detection on RunPod GPU.

Uses Emine's 14k drone dataset + COCO negatives.
Exports to ONNX for Luxonis converter.

Prerequisites:
  - Dataset uploaded to RunPod at ~/dataset/ (YOLO format: images/ + labels/)
  - GPU instance with PyTorch + CUDA

Usage:
    python train_yolov5n.py                      # Full training
    python train_yolov5n.py --epochs 50          # Quick test
    python train_yolov5n.py --export-only        # Just export existing weights
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path(os.path.expanduser("~/dataset"))
YOLOV5_DIR = Path("yolov5")
IMG_SIZE = 416
EPOCHS = 150


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd[:8])}")
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def setup_yolov5():
    if YOLOV5_DIR.exists():
        print("YOLOv5 already cloned")
        return
    print("Cloning YOLOv5...")
    run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
    run([sys.executable, "-m", "pip", "install", "-r", str(YOLOV5_DIR / "requirements.txt")])


def write_data_yaml():
    yaml_path = DATASET_DIR / "data.yaml"

    train_path = (DATASET_DIR / "images" / "train").resolve()
    val_path = (DATASET_DIR / "images" / "val").resolve()

    yaml_path.write_text(
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"nc: 1\n"
        f"names: ['drone']\n"
    )
    print(f"Data config: {yaml_path}")
    return yaml_path


def train(epochs, batch_size):
    data_yaml = write_data_yaml()

    train_count = len(list((DATASET_DIR / "images" / "train").glob("*")))
    val_count = len(list((DATASET_DIR / "images" / "val").glob("*")))
    print(f"\nTraining YOLOv5n — {epochs} epochs, {IMG_SIZE}x{IMG_SIZE}, batch {batch_size}")
    print(f"Dataset: {train_count} train / {val_count} val")
    print("=" * 60)

    run([
        sys.executable, "train.py",
        "--img", str(IMG_SIZE),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", str(data_yaml.resolve()),
        "--weights", "yolov5n.pt",  # pretrained nano weights from Ultralytics
        "--name", "drone_v5n",
        "--device", "0",
        "--project", "runs/train",
    ], cwd=str(YOLOV5_DIR))


def export_onnx():
    weights = YOLOV5_DIR / "runs" / "train" / "drone_v5n" / "weights" / "best.pt"
    if not weights.exists():
        print(f"ERROR: Weights not found at {weights}")
        sys.exit(1)

    print(f"\nExporting {weights} to ONNX...")
    run([
        sys.executable, "export.py",
        "--weights", str(weights.resolve()),
        "--img-size", str(IMG_SIZE), str(IMG_SIZE),
        "--include", "onnx",
        "--simplify",
        "--opset", "12",
    ], cwd=str(YOLOV5_DIR))

    onnx_path = weights.with_suffix(".onnx")
    if onnx_path.exists():
        dest = Path("drone_yolov5n.onnx")
        onnx_path.rename(dest)
        print(f"ONNX saved: {dest}")
        return dest
    else:
        print("ONNX export failed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv5n drone detection")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    if args.export_only:
        export_onnx()
        print("\nUpload drone_yolov5n.onnx to tools.luxonis.com to convert to NNArchive")
        return

    # Check dataset
    train_dir = DATASET_DIR / "images" / "train"
    if not train_dir.exists() or len(list(train_dir.glob("*"))) == 0:
        print(f"ERROR: No dataset at {DATASET_DIR}")
        print("Upload dataset: scp -r dataset/ root@<runpod>:~/")
        sys.exit(1)

    setup_yolov5()
    train(args.epochs, args.batch)
    onnx_path = export_onnx()

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  ONNX: {onnx_path}")
    print(f"\nNext: upload {onnx_path} to tools.luxonis.com")
    print(f"  Platform: RVC2, Shape: [1,3,{IMG_SIZE},{IMG_SIZE}]")
    print(f"  Class names: drone")
    print(f"  Output: NN Archive")


if __name__ == "__main__":
    main()
