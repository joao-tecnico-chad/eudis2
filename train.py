"""
Train YOLOv6n on Zhejiang University Drone Detection Dataset
Dataset: https://universe.roboflow.com/zhejiang-university-china-dliq1/drones-detection-with-yolov8

Requirements:
    pip install roboflow supervision

Setup:
    1. Get your Roboflow API key from https://app.roboflow.com/settings/api
    2. Set it as an environment variable:
         export ROBOFLOW_API_KEY=your_key_here
    3. Run: python train.py

Output:
    YOLOv6/runs/train/exp/weights/best.pt
"""

import os
import subprocess
import sys
from pathlib import Path


ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
DATASET_DIR = Path("dataset")
YOLOV6_DIR = Path("YOLOv6")
IMG_SIZE = 416
BATCH_SIZE = 16
EPOCHS = 50


def download_dataset() -> Path:
    if not ROBOFLOW_API_KEY:
        print("ERROR: ROBOFLOW_API_KEY environment variable not set.")
        print("Get your key at https://app.roboflow.com/settings/api")
        print("Then run: export ROBOFLOW_API_KEY=your_key_here")
        sys.exit(1)

    from roboflow import Roboflow

    print("Downloading Zhejiang drone dataset from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("zhejiang-university-china-dliq1").project("drones-detection-with-yolov8")
    version = project.version(1)
    dataset = version.download("yolov5", location=str(DATASET_DIR))
    return Path(dataset.location)


def setup_yolov6() -> None:
    if YOLOV6_DIR.exists():
        print("YOLOv6 repo already exists, skipping clone.")
        return

    print("Cloning YOLOv6 repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/meituan/YOLOv6.git"],
        check=True,
    )

    print("Installing YOLOv6 requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(YOLOV6_DIR / "requirements.txt")],
        check=True,
    )


def write_data_yaml(dataset_path: Path) -> Path:
    yaml_path = YOLOV6_DIR / "data" / "drone.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = (dataset_path / "train" / "images").resolve()
    val_path = (dataset_path / "valid" / "images").resolve()

    yaml_path.write_text(
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"nc: 1\n"
        f"names: ['drone']\n"
    )
    print(f"Dataset config written to {yaml_path}")
    return yaml_path


def train(data_yaml: Path) -> None:
    print(f"\nStarting YOLOv6n training — {EPOCHS} epochs, {IMG_SIZE}x{IMG_SIZE}, batch {BATCH_SIZE}")
    subprocess.run(
        [
            sys.executable, "tools/train.py",
            "--batch", str(BATCH_SIZE),
            "--conf", "configs/yolov6n.py",
            "--data", str(data_yaml.resolve()),
            "--img-size", str(IMG_SIZE),
            "--epochs", str(EPOCHS),
            "--name", "drone_yolov6n",
            "--device", "0",  # use GPU if available; set to "cpu" to force CPU
        ],
        cwd=str(YOLOV6_DIR),
        check=True,
    )


def main() -> None:
    dataset_path = download_dataset()
    setup_yolov6()
    data_yaml = write_data_yaml(dataset_path)
    train(data_yaml)

    weights = YOLOV6_DIR / "runs" / "train" / "drone_yolov6n" / "weights" / "best.pt"
    if weights.exists():
        print(f"\nTraining complete. Weights saved to: {weights}")
        print("Next step: run convert_to_blob.py to prepare model for OAK-1W")
    else:
        print("\nTraining finished. Check YOLOv6/runs/train/drone_yolov6n/weights/ for output.")


if __name__ == "__main__":
    main()
