"""
Download, validate, and merge drone detection datasets for YOLOv6 training.

Datasets (images OF drones, not aerial imagery):
  1. DroneNet (GitHub)          — 2,664 labeled DJI drone images
  2. Roboflow Zhejiang Uni      — 4,231 drone images (needs ROBOFLOW_API_KEY)
  3. Kaggle drone-dataset-uav   — 1,359 rotary-wing UAV images
  4. Kaggle yolo-drone-detection — 1,012 + 347 drone images

Negative samples (reduce false positives):
  5. COCO birds + airplanes     — ~3,000 confuser images (empty drone labels)
  6. COCO outdoor backgrounds   — ~1,000 sky/outdoor scenes (empty labels)

Total target: ~12,000+ images with proper train/val split.

Usage:
    export ROBOFLOW_API_KEY=your_key    # optional, skips Roboflow if not set
    python tools/prepare_dataset.py
    python tools/prepare_dataset.py --validate   # spot-check label quality
    python tools/prepare_dataset.py --stats      # print dataset statistics
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path("datasets")
MERGED_DIR = DATASET_DIR / "merged"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

random.seed(42)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(cmd[:6])}...")
    subprocess.run(cmd, check=True, **kwargs)


def download_dronenet() -> Path:
    """DroneNet — 2,664 labeled DJI drone images from GitHub."""
    dest = DATASET_DIR / "dronenet"
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"[dronenet] Already downloaded.")
        return dest
    print("[dronenet] Cloning from GitHub...")
    dest.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth=1", "https://github.com/chuanenlin/drone-net.git",
         str(dest / "repo")])
    return dest


def download_roboflow() -> Path | None:
    """Roboflow Zhejiang Uni — 4,231 drone images."""
    dest = DATASET_DIR / "roboflow-drones"
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"[roboflow] Already downloaded.")
        return dest
    if not ROBOFLOW_API_KEY:
        print("[roboflow] Skipped — ROBOFLOW_API_KEY not set.")
        return None
    print("[roboflow] Downloading via Roboflow SDK...")
    from roboflow import Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("zhejiang-university-china-dliq1").project("drones-detection-with-yolov8")
    project.version(1).download("yolov5", location=str(dest))
    return dest


def download_kaggle(slug: str, name: str) -> Path:
    """Download a Kaggle dataset."""
    dest = DATASET_DIR / name
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"[{name}] Already downloaded.")
        return dest
    print(f"[{name}] Downloading from Kaggle...")
    dest.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip", "-q"])
    return dest


def download_coco_negatives() -> Path:
    """Download COCO val2017 and extract birds, airplanes, and outdoor scenes as negatives."""
    dest = DATASET_DIR / "coco-negatives"
    if dest.exists() and any(dest.glob("*.jpg")):
        count = len(list(dest.glob("*.jpg")))
        print(f"[coco-negatives] Already prepared ({count} images).")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    coco_dir = DATASET_DIR / "coco"

    # Download annotations
    ann_file = coco_dir / "annotations" / "instances_val2017.json"
    if not ann_file.exists():
        print("[coco-negatives] Downloading COCO val2017 annotations...")
        coco_dir.mkdir(parents=True, exist_ok=True)
        run(["wget", "-q", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
             "-O", str(coco_dir / "annotations.zip")])
        run(["unzip", "-q", "-o", str(coco_dir / "annotations.zip"), "-d", str(coco_dir)])

    # Download val2017 images
    img_dir = coco_dir / "val2017"
    if not img_dir.exists():
        print("[coco-negatives] Downloading COCO val2017 images...")
        run(["wget", "-q", "http://images.cocodataset.org/zips/val2017.zip",
             "-O", str(coco_dir / "val2017.zip")])
        run(["unzip", "-q", "-o", str(coco_dir / "val2017.zip"), "-d", str(coco_dir)])

    # Parse annotations — find images with birds (16), airplanes (5), kites (38)
    # and outdoor scenes without these (pure backgrounds)
    print("[coco-negatives] Extracting confuser and background images...")
    with open(ann_file) as f:
        coco = json.load(f)

    confuser_cats = {16, 5, 38}  # bird, airplane, kite
    outdoor_cats = {
        1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 58, 59, 60, 61, 62, 63
    }  # outdoor-related COCO categories

    img_cats = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        img_cats.setdefault(img_id, set()).add(cat_id)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    confuser_imgs = []
    background_imgs = []
    for img_id, cats in img_cats.items():
        if cats & confuser_cats:
            confuser_imgs.append(id_to_file[img_id])
        elif cats & outdoor_cats:
            background_imgs.append(id_to_file[img_id])

    # Take up to 2000 confusers and 1000 backgrounds
    random.shuffle(confuser_imgs)
    random.shuffle(background_imgs)
    selected = confuser_imgs[:2000] + background_imgs[:1000]

    for fname in selected:
        src = img_dir / fname
        if src.exists():
            shutil.copy(src, dest / fname)

    print(f"[coco-negatives] Extracted {len(list(dest.glob('*.jpg')))} images "
          f"({min(len(confuser_imgs), 2000)} confusers + {min(len(background_imgs), 1000)} backgrounds)")
    return dest


# ---------------------------------------------------------------------------
# Dataset merging
# ---------------------------------------------------------------------------

def find_image_label_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    """Find all image files with their corresponding label files."""
    pairs = []
    for img in sorted(root.rglob("*.jpg")) + sorted(root.rglob("*.png")):
        # Skip hidden/cache dirs
        if any(p.startswith(".") for p in img.parts):
            continue
        # Look for label in parallel labels/ dir or same dir
        stem = img.stem
        for lbl_dir in [img.parent.parent / "labels", img.parent / "labels", img.parent]:
            lbl = lbl_dir / f"{stem}.txt"
            if lbl.exists() and lbl != img:
                pairs.append((img, lbl))
                break
        else:
            pairs.append((img, None))
    return pairs


def image_hash(path: Path) -> str:
    """Quick hash for deduplication (first 8KB)."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read(8192)).hexdigest()


def validate_label(lbl_path: Path) -> bool:
    """Check if a YOLO label file is valid."""
    try:
        for line in lbl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            cls = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            # Check normalized coords are in valid range
            if any(c < -0.5 or c > 1.5 for c in coords):
                return False
        return True
    except (ValueError, IndexError):
        return False


def merge_all(sources: dict[str, Path | None]) -> None:
    """Merge all datasets into MERGED_DIR with deduplication."""
    if MERGED_DIR.exists() and any((MERGED_DIR / "images" / "train").glob("*")):
        count = len(list((MERGED_DIR / "images" / "train").glob("*")))
        print(f"\nMerged dataset already exists ({count} train images). Delete datasets/merged/ to re-merge.")
        return

    TRAIN_IMG = MERGED_DIR / "images" / "train"
    VAL_IMG = MERGED_DIR / "images" / "val"
    TRAIN_LBL = MERGED_DIR / "labels" / "train"
    VAL_LBL = MERGED_DIR / "labels" / "val"
    for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
        p.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    total_added = 0
    total_skipped_dup = 0
    total_skipped_bad = 0

    for name, src in sources.items():
        if src is None:
            continue
        print(f"\nProcessing [{name}]...")
        pairs = find_image_label_pairs(src)
        added = 0

        for img, lbl in pairs:
            # Dedup
            h = image_hash(img)
            if h in seen_hashes:
                total_skipped_dup += 1
                continue
            seen_hashes.add(h)

            # Validate label
            if lbl is not None and not validate_label(lbl):
                total_skipped_bad += 1
                continue

            # Prefix to avoid name collisions
            prefix = name.replace("-", "_")
            dest_name = f"{prefix}_{img.name}"
            lbl_name = f"{prefix}_{img.stem}.txt"

            # 90/10 split
            is_train = random.random() < 0.9
            di = TRAIN_IMG if is_train else VAL_IMG
            dl = TRAIN_LBL if is_train else VAL_LBL

            shutil.copy(img, di / dest_name)
            if lbl is not None:
                # Remap all classes to 0 (drone)
                lines = lbl.read_text().strip().split("\n")
                remapped = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"  # force class 0
                        remapped.append(" ".join(parts))
                (dl / lbl_name).write_text("\n".join(remapped) + "\n")
            else:
                # Negative sample — empty label
                (dl / lbl_name).touch()

            added += 1

        total_added += added
        print(f"  Added {added} images from [{name}]")

    print(f"\n{'='*50}")
    print(f"Total: {total_added} images merged")
    print(f"Duplicates removed: {total_skipped_dup}")
    print(f"Bad labels skipped: {total_skipped_bad}")
    print(f"Train: {len(list(TRAIN_IMG.glob('*')))}  Val: {len(list(VAL_IMG.glob('*')))}")


def write_data_yaml() -> Path:
    """Write YOLO dataset config."""
    yaml_path = MERGED_DIR / "data.yaml"
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


# ---------------------------------------------------------------------------
# Validation / stats
# ---------------------------------------------------------------------------

def print_stats() -> None:
    """Print dataset statistics."""
    for split in ["train", "val"]:
        img_dir = MERGED_DIR / "images" / split
        lbl_dir = MERGED_DIR / "labels" / split
        if not img_dir.exists():
            print(f"  {split}: not found")
            continue
        imgs = list(img_dir.glob("*"))
        lbls = list(lbl_dir.glob("*.txt"))
        empty = sum(1 for l in lbls if l.stat().st_size == 0)
        with_drones = len(lbls) - empty
        print(f"  {split}: {len(imgs)} images, {with_drones} with drones, {empty} negatives")

    # Source breakdown
    print("\nBy source:")
    for prefix in ["dronenet", "roboflow", "drone_uav", "yolo_drone", "coco"]:
        count = len(list((MERGED_DIR / "images" / "train").glob(f"{prefix}*")))
        count += len(list((MERGED_DIR / "images" / "val").glob(f"{prefix}*")))
        if count > 0:
            print(f"  {prefix}: {count}")


def spot_check(n: int = 10) -> None:
    """Randomly sample and print label info for validation."""
    lbls = list((MERGED_DIR / "labels" / "train").glob("*.txt"))
    if not lbls:
        print("No labels found.")
        return
    sample = random.sample(lbls, min(n, len(lbls)))
    print(f"\nSpot-checking {len(sample)} labels:")
    for lbl in sample:
        content = lbl.read_text().strip()
        n_boxes = len(content.split("\n")) if content else 0
        status = "OK" if content else "NEGATIVE"
        if content:
            # Check first box
            parts = content.split("\n")[0].split()
            if len(parts) == 5:
                cx, cy, w, h = [float(x) for x in parts[1:]]
                area_pct = w * h * 100
                status = f"OK ({n_boxes} box{'es' if n_boxes > 1 else ''}, largest ~{area_pct:.1f}% of frame)"
        print(f"  {lbl.name}: {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare drone detection dataset")
    parser.add_argument("--validate", action="store_true", help="Spot-check label quality")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--no-negatives", action="store_true", help="Skip negative samples")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return
    if args.validate:
        spot_check(20)
        return

    print("=== Drone Detection Dataset Preparation ===\n")

    # Download all sources
    sources = {}
    sources["dronenet"] = download_dronenet()
    sources["roboflow"] = download_roboflow()
    sources["drone_uav"] = download_kaggle("dasmehdixtr/drone-dataset-uav", "drone-uav")
    sources["yolo_drone"] = download_kaggle("muki2003/yolo-drone-detection-dataset", "yolo-drone")

    if not args.no_negatives:
        sources["coco_neg"] = download_coco_negatives()

    # Merge with dedup and validation
    merge_all(sources)
    write_data_yaml()

    print("\n=== Done ===")
    print_stats()
    print(f"\nReady for training. Dataset at: {MERGED_DIR}")
    print(f"Config at: {MERGED_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
