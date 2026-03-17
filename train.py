"""
Train YOLOv6 drone detection locally on RTX 3080 (10GB)

Supports both YOLOv6n (nano) and YOLOv6s (small) with the enhanced
merged dataset (Roboflow + 2 Kaggle sets).

Usage:
    python train.py                  # Train v6n (default)
    python train.py --model v6s      # Train v6s
    python train.py --resume         # Resume last interrupted training
    python train.py --model v6s --resume
    python train.py --clean          # Re-merge dataset (includes negatives)

Stop anytime with Ctrl+C — training auto-saves checkpoints every epoch.
Resume later with --resume and it picks up where it left off.

Setup:
    1. export ROBOFLOW_API_KEY=your_key_here
    2. pip install -r requirements.txt
    3. python train.py
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "gZGnOpNOYEeRXNJiMkAJ")
DATASET_DIR = Path("datasets")
MERGED_DIR = DATASET_DIR / "merged"
YOLOV6_DIR = Path("YOLOv6")
IMG_SIZE = 416
EPOCHS = 100

NEGATIVE_DATASET = "imsparsh/coco-person-only-detection"
NEGATIVE_DIR = DATASET_DIR / "negatives"
NEGATIVE_COUNT = 800

MODELS = {
    "v6n": {"conf": "configs/yolov6n.py", "batch": 32, "name": "drone_yolov6n"},
    "v6s": {"conf": "configs/yolov6s.py", "batch": 16, "name": "drone_yolov6s"},
}


def download_datasets() -> None:
    """Download all three datasets if not already present."""
    if not ROBOFLOW_API_KEY:
        print("ERROR: ROBOFLOW_API_KEY not set.")
        print("Get your key at https://app.roboflow.com/settings/api")
        print("Then: export ROBOFLOW_API_KEY=your_key_here")
        sys.exit(1)

    DATASET_DIR.mkdir(exist_ok=True)

    # 1. Roboflow
    rf_dir = DATASET_DIR / "roboflow-drones"
    if not rf_dir.exists():
        from roboflow import Roboflow
        print("Downloading Roboflow dataset (Zhejiang University)...")
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("zhejiang-university-china-dliq1").project("drones-detection-with-yolov8")
        project.version(1).download("yolov5", location=str(rf_dir))
    else:
        print("Roboflow dataset already downloaded.")

    # 2. Kaggle: drone-dataset-uav
    uav_dir = DATASET_DIR / "drone-uav"
    if not uav_dir.exists():
        print("Downloading Kaggle drone-dataset-uav...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "dasmehdixtr/drone-dataset-uav",
             "-p", str(uav_dir), "--unzip", "-q"],
            check=True,
        )
    else:
        print("Kaggle drone-uav dataset already downloaded.")

    # 3. Kaggle: yolo-drone-detection-dataset
    yolo_dir = DATASET_DIR / "yolo-drone"
    if not yolo_dir.exists():
        print("Downloading Kaggle yolo-drone-detection-dataset...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "muki2003/yolo-drone-detection-dataset",
             "-p", str(yolo_dir), "--unzip", "-q"],
            check=True,
        )
    else:
        print("Kaggle yolo-drone dataset already downloaded.")


def download_negatives() -> bool:
    """Download COCO person images as negative samples (no drones).

    Returns True if negatives are available, False otherwise.
    """
    if NEGATIVE_DIR.exists() and any(NEGATIVE_DIR.glob("*.jpg")):
        count = len(list(NEGATIVE_DIR.glob("*.jpg")) + list(NEGATIVE_DIR.glob("*.png")))
        print(f"Negative samples already downloaded ({count} images).")
        return True

    print(f"Downloading negative samples from {NEGATIVE_DATASET}...")
    try:
        NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", NEGATIVE_DATASET,
             "-p", str(NEGATIVE_DIR), "--unzip", "-q"],
            check=True,
        )
        # Count what we got
        images = list(NEGATIVE_DIR.rglob("*.jpg")) + list(NEGATIVE_DIR.rglob("*.png"))
        print(f"Downloaded {len(images)} negative sample images.")
        return len(images) > 0
    except Exception as e:
        print(f"WARNING: Failed to download negatives: {e}")
        print("Training will continue without negative samples.")
        return False


def merge_datasets(include_negatives: bool = False) -> None:
    """Merge all datasets into a single YOLO-format directory."""
    if MERGED_DIR.exists() and any((MERGED_DIR / "images" / "train").glob("*")):
        count = len(list((MERGED_DIR / "images" / "train").glob("*")))
        print(f"Merged dataset already exists ({count} train images). Skipping merge.")
        return

    TRAIN_IMG = MERGED_DIR / "images" / "train"
    VAL_IMG = MERGED_DIR / "images" / "val"
    TRAIN_LBL = MERGED_DIR / "labels" / "train"
    VAL_LBL = MERGED_DIR / "labels" / "val"

    for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

    def copy_split(src_img, src_lbl, dst_img, dst_lbl, split_ratio=0.9):
        images = list(Path(src_img).glob("*.jpg")) + list(Path(src_img).glob("*.png"))
        n_train = int(len(images) * split_ratio)
        for i, img in enumerate(images):
            di = dst_img if i < n_train else VAL_IMG
            dl = dst_lbl if i < n_train else VAL_LBL
            shutil.copy(img, di / img.name)
            lbl = Path(src_lbl) / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, dl / lbl.name)

    # 1. Roboflow (already split into train/valid)
    rf = DATASET_DIR / "roboflow-drones"
    for img in (rf / "train" / "images").glob("*"):
        shutil.copy(img, TRAIN_IMG / img.name)
    for lbl in (rf / "train" / "labels").glob("*"):
        shutil.copy(lbl, TRAIN_LBL / lbl.name)
    for img in (rf / "valid" / "images").glob("*"):
        shutil.copy(img, VAL_IMG / img.name)
    for lbl in (rf / "valid" / "labels").glob("*"):
        shutil.copy(lbl, VAL_LBL / lbl.name)
    print(f"Roboflow: {len(list(TRAIN_IMG.glob('*')))} train, {len(list(VAL_IMG.glob('*')))} val")

    # 2. drone-dataset-uav
    for img_dir in list((DATASET_DIR / "drone-uav").glob("**/images"))[:3]:
        lbl_dir = img_dir.parent / "labels"
        if lbl_dir.exists():
            copy_split(img_dir, lbl_dir, TRAIN_IMG, TRAIN_LBL)
    print(f"After UAV: {len(list(TRAIN_IMG.glob('*')))} train, {len(list(VAL_IMG.glob('*')))} val")

    # 3. yolo-drone-detection-dataset
    for img_dir in list((DATASET_DIR / "yolo-drone").glob("**/images"))[:3]:
        lbl_dir = img_dir.parent / "labels"
        if lbl_dir.exists():
            copy_split(img_dir, lbl_dir, TRAIN_IMG, TRAIN_LBL)
    print(f"After drones: {len(list(TRAIN_IMG.glob('*')))} train, {len(list(VAL_IMG.glob('*')))} val")

    # 4. Negative samples (background images with empty labels)
    if include_negatives:
        neg_images = list(NEGATIVE_DIR.rglob("*.jpg")) + list(NEGATIVE_DIR.rglob("*.png"))
        random.seed(42)
        random.shuffle(neg_images)
        neg_images = neg_images[:NEGATIVE_COUNT]

        n_train = int(len(neg_images) * 0.9)
        for i, img in enumerate(neg_images):
            if i < n_train:
                dst_img, dst_lbl = TRAIN_IMG, TRAIN_LBL
            else:
                dst_img, dst_lbl = VAL_IMG, VAL_LBL
            dest_name = f"neg_{img.stem}{img.suffix}"
            shutil.copy(img, dst_img / dest_name)
            (dst_lbl / f"neg_{img.stem}.txt").touch()

        print(f"Added {len(neg_images)} negative samples ({n_train} train, {len(neg_images) - n_train} val)")

    print(f"Final: {len(list(TRAIN_IMG.glob('*')))} train, {len(list(VAL_IMG.glob('*')))} val")


def setup_yolov6() -> None:
    if YOLOV6_DIR.exists():
        print("YOLOv6 repo already exists, skipping clone.")
        return

    print("Cloning YOLOv6...")
    subprocess.run(["git", "clone", "https://github.com/meituan/YOLOv6.git"], check=True)
    print("Installing YOLOv6 requirements...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(YOLOV6_DIR / "requirements.txt")],
        check=True,
    )


def write_data_yaml() -> Path:
    yaml_path = YOLOV6_DIR / "data" / "drones_merged.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = (MERGED_DIR / "images" / "train").resolve()
    val_path = (MERGED_DIR / "images" / "val").resolve()

    yaml_path.write_text(
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"nc: 1\n"
        f"names: ['drone']\n"
    )
    print(f"Dataset config: {yaml_path}")
    return yaml_path


def train(model: str, data_yaml: Path, resume: bool) -> None:
    cfg = MODELS[model]

    cmd = [
        sys.executable, "tools/train.py",
        "--batch", str(cfg["batch"]),
        "--conf", cfg["conf"],
        "--data", str(data_yaml.resolve()),
        "--img-size", str(IMG_SIZE),
        "--epochs", str(EPOCHS),
        "--name", cfg["name"],
        "--device", "0",
    ]

    # Resume from last checkpoint if requested
    if resume:
        ckpt = YOLOV6_DIR / "runs" / "train" / cfg["name"] / "weights" / "last_ckpt.pt"
        if ckpt.exists():
            cmd.extend(["--resume", str(ckpt.resolve())])
            print(f"Resuming from {ckpt}")
        else:
            print(f"No checkpoint found at {ckpt}, starting fresh.")

    print(f"\nTraining YOLOv6{model[1:]} — {EPOCHS} epochs, {IMG_SIZE}x{IMG_SIZE}, batch {cfg['batch']}")
    print("Press Ctrl+C to stop. Resume later with: python train.py --resume")
    print("-" * 60)

    subprocess.run(cmd, cwd=str(YOLOV6_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv6 drone detection locally")
    parser.add_argument("--model", choices=["v6n", "v6s"], default="v6n",
                        help="Model variant: v6n (nano/fast) or v6s (small/accurate)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--clean", action="store_true",
                        help="Delete merged dataset and re-merge (use when adding negatives)")
    args = parser.parse_args()

    if args.clean and MERGED_DIR.exists():
        print("Cleaning merged dataset...")
        shutil.rmtree(MERGED_DIR)

    download_datasets()
    has_negatives = download_negatives()
    merge_datasets(include_negatives=has_negatives)
    setup_yolov6()
    data_yaml = write_data_yaml()
    train(args.model, data_yaml, args.resume)

    cfg = MODELS[args.model]
    weights = YOLOV6_DIR / "runs" / "train" / cfg["name"] / "weights" / "best_ckpt.pt"
    if weights.exists():
        print(f"\nDone! Best weights: {weights}")
        print("Next: python convert_to_blob.py")
    else:
        print(f"\nCheck YOLOv6/runs/train/{cfg['name']}/weights/ for output.")


if __name__ == "__main__":
    main()
