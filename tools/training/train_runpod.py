"""
Train YOLOv6n on RunPod GPU instance.

This script runs ON the RunPod machine. Steps:
  1. Downloads COCO negatives (birds, planes, outdoor) for false positive reduction
  2. Merges negatives into the drone dataset
  3. Clones YOLOv6 and trains for 150 epochs at 416x416
  4. Exports best weights to ONNX
  5. Converts to OAK blob (FP16, 6 shaves)

Prerequisites:
  - Dataset uploaded to RunPod at ~/dataset/ (images/ and labels/ dirs)
  - GPU instance with PyTorch + CUDA

Usage:
    python train_runpod.py                    # Full pipeline
    python train_runpod.py --skip-negatives   # Skip COCO download
    python train_runpod.py --epochs 50        # Fewer epochs for testing
    python train_runpod.py --export-only      # Just export existing weights
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path(os.path.expanduser("~/dataset"))
YOLOV6_DIR = Path("YOLOv6")
IMG_SIZE = 416
SHAVES = 6

random.seed(42)


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd[:8])}")
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


# ---------------------------------------------------------------------------
# COCO negatives
# ---------------------------------------------------------------------------

def download_coco_negatives():
    """Download COCO val2017 and extract birds, planes, kites, outdoor scenes."""
    neg_dir = DATASET_DIR / "negatives"
    if neg_dir.exists() and len(list(neg_dir.glob("*.jpg"))) > 500:
        print(f"Negatives already exist ({len(list(neg_dir.glob('*.jpg')))} images)")
        return

    neg_dir.mkdir(parents=True, exist_ok=True)
    coco_dir = Path("/tmp/coco")
    coco_dir.mkdir(exist_ok=True)

    # Download
    ann_zip = coco_dir / "annotations.zip"
    img_zip = coco_dir / "val2017.zip"

    if not (coco_dir / "annotations" / "instances_val2017.json").exists():
        print("Downloading COCO val2017 annotations...")
        run(["wget", "-q", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
             "-O", str(ann_zip)])
        run(["unzip", "-q", "-o", str(ann_zip), "-d", str(coco_dir)])

    if not (coco_dir / "val2017").exists():
        print("Downloading COCO val2017 images (1GB)...")
        run(["wget", "-q", "http://images.cocodataset.org/zips/val2017.zip",
             "-O", str(img_zip)])
        run(["unzip", "-q", "-o", str(img_zip), "-d", str(coco_dir)])

    # Parse annotations
    print("Extracting negative samples...")
    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    with open(ann_file) as f:
        coco = json.load(f)

    # Categories: bird=16, airplane=5, kite=38
    confuser_cats = {16, 5, 38}
    # Outdoor categories for background images
    outdoor_cats = {1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}

    img_cats = {}
    for ann in coco["annotations"]:
        img_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    confusers = [id_to_file[i] for i, cats in img_cats.items() if cats & confuser_cats]
    backgrounds = [id_to_file[i] for i, cats in img_cats.items()
                   if (cats & outdoor_cats) and not (cats & confuser_cats)]

    random.shuffle(confusers)
    random.shuffle(backgrounds)
    selected = confusers[:2000] + backgrounds[:1000]

    img_dir = coco_dir / "val2017"
    count = 0
    for fname in selected:
        src = img_dir / fname
        if src.exists():
            shutil.copy(src, neg_dir / fname)
            count += 1

    print(f"Extracted {count} negatives ({min(len(confusers), 2000)} confusers + {min(len(backgrounds), 1000)} backgrounds)")


def add_negatives_to_dataset():
    """Add COCO negatives to the training dataset with empty labels."""
    neg_dir = DATASET_DIR / "negatives"
    if not neg_dir.exists():
        return

    train_img = DATASET_DIR / "images" / "train"
    train_lbl = DATASET_DIR / "labels" / "train"
    val_img = DATASET_DIR / "images" / "val"
    val_lbl = DATASET_DIR / "labels" / "val"

    negs = list(neg_dir.glob("*.jpg"))
    random.shuffle(negs)

    added = 0
    for img in negs:
        is_val = random.random() < 0.1
        di = val_img if is_val else train_img
        dl = val_lbl if is_val else train_lbl

        dest = f"neg_{img.name}"
        if not (di / dest).exists():
            shutil.copy(img, di / dest)
            (dl / f"neg_{img.stem}.txt").touch()  # empty = no drone
            added += 1

    print(f"Added {added} negative samples to dataset")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def setup_yolov6():
    if YOLOV6_DIR.exists():
        print("YOLOv6 already cloned")
        return
    print("Cloning YOLOv6...")
    run(["git", "clone", "https://github.com/meituan/YOLOv6.git"])
    run([sys.executable, "-m", "pip", "install", "-r", str(YOLOV6_DIR / "requirements.txt")])


def write_data_yaml():
    yaml_path = YOLOV6_DIR / "data" / "drones.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

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


def train(epochs, batch_size=32):
    data_yaml = write_data_yaml()

    cmd = [
        sys.executable, "tools/train.py",
        "--batch", str(batch_size),
        "--conf", "configs/yolov6n.py",
        "--data", str(data_yaml.resolve()),
        "--img-size", str(IMG_SIZE),
        "--epochs", str(epochs),
        "--name", "drone_v6n",
        "--device", "0",
    ]

    train_count = len(list((DATASET_DIR / "images" / "train").glob("*")))
    val_count = len(list((DATASET_DIR / "images" / "val").glob("*")))
    print(f"\nTraining YOLOv6n — {epochs} epochs, {IMG_SIZE}x{IMG_SIZE}, batch {batch_size}")
    print(f"Dataset: {train_count} train / {val_count} val")
    print("=" * 60)

    run(cmd, cwd=str(YOLOV6_DIR))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx():
    weights = YOLOV6_DIR / "runs" / "train" / "drone_v6n" / "weights" / "best_ckpt.pt"
    if not weights.exists():
        print(f"ERROR: Weights not found at {weights}")
        sys.exit(1)

    onnx_path = Path("drone_yolov6n.onnx")
    print(f"\nExporting {weights} to ONNX...")

    run([
        sys.executable, "deploy/ONNX/export_onnx.py",
        "--weights", str(weights.resolve()),
        "--img-size", str(IMG_SIZE), str(IMG_SIZE),
        "--batch-size", "1",
        "--simplify",
    ], cwd=str(YOLOV6_DIR))

    # YOLOv6 saves as best_ckpt.onnx next to the weights
    exported = weights.parent / "best_ckpt.onnx"
    if exported.exists():
        shutil.move(str(exported), str(onnx_path))
    print(f"ONNX saved: {onnx_path}")
    return onnx_path


def convert_to_blob(onnx_path):
    print(f"\nConverting to OAK blob (FP16, {SHAVES} shaves)...")

    try:
        import blobconverter
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "blobconverter"])
        import blobconverter

    blob_path = blobconverter.from_onnx(
        model=str(onnx_path),
        data_type="FP16",
        shaves=SHAVES,
        version="2022.1",
        output_dir=".",
        optimizer_params=[
            f"--input_shape=[1,3,{IMG_SIZE},{IMG_SIZE}]",
            "--data_type=FP16",
        ],
    )

    final = Path("drone_yolov6n.blob")
    Path(blob_path).rename(final)
    print(f"Blob saved: {final} ({final.stat().st_size // 1024}KB)")
    return final


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv6n on RunPod")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=64, help="Batch size (64 for RTX 5090 32GB)")
    parser.add_argument("--skip-negatives", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    # Check dataset exists
    train_dir = DATASET_DIR / "images" / "train"
    if not train_dir.exists() or len(list(train_dir.glob("*"))) == 0:
        print(f"ERROR: No dataset found at {DATASET_DIR}")
        print("Upload your dataset first: rsync -avz final_dataset/ runpod:~/dataset/")
        sys.exit(1)

    if args.export_only:
        onnx = export_onnx()
        convert_to_blob(onnx)
        return

    # Add negatives
    if not args.skip_negatives:
        download_coco_negatives()
        add_negatives_to_dataset()

    # Train
    setup_yolov6()
    train(args.epochs, args.batch)

    # Export
    onnx = export_onnx()
    convert_to_blob(onnx)

    print("\n" + "=" * 60)
    print("DONE! Files ready:")
    print(f"  ONNX:  drone_yolov6n.onnx")
    print(f"  Blob:  drone_yolov6n.blob")
    print(f"\nCopy to your project:")
    print(f"  scp drone_yolov6n.blob your-mac:~/Dev/eudis2/models/")
    print(f"  scp drone_yolov6n.onnx your-mac:~/Dev/eudis2/models/")


if __name__ == "__main__":
    main()
