# Drone Detection with YOLOv6

Real-time drone detection for OAK-1W edge device.

## Project Overview

- **Model**: YOLOv6n / YOLOv6s
- **Target Device**: OAK-1W (416x416, FP16)
- **Dataset**: Merged drone datasets (~15k+ images)
- **Performance**: mAP@0.5 = 91.9% (v6n, base dataset)

## Project Structure

```
eudis2/
├── kaggle/
│   ├── train_kaggle_v6n.ipynb    # YOLOv6n training notebook
│   ├── train_kaggle_v6s.ipynb    # YOLOv6s training notebook
│   ├── train_colab_v6n.ipynb     # Colab version
│   └── train_colab_v6s.ipynb     # Colab version
├── detect_drone_mac.py           # Local inference (macOS)
├── detect_drone_oak.py           # OAK-1W inference
├── convert_to_blob.py            # ONNX to OAK blob converter
├── train.py                      # Training script reference
└── requirements.txt
```

## Training (Kaggle)

1. Open `train_kaggle_v6n.ipynb` or `train_kaggle_v6s.ipynb` in Kaggle
2. Enable GPU accelerator (T4 P100)
3. Run all cells
4. Download `best_ckpt.pt` and `.blob` from output

### Models

| Model | Size | Speed (OAK) | Accuracy | Use Case |
|-------|------|-------------|----------|----------|
| YOLOv6n | 10MB | ~30 FPS | Good | Real-time detection |
| YOLOv6s | 20MB | ~15 FPS | Better | Higher accuracy needed |

## Local Inference

```bash
# Install dependencies
pip install onnxruntime opencv-python numpy

# Run on webcam
python detect_drone_mac.py --source 0

# Run on image
python detect_drone_mac.py --source image.jpg

# Run on video
python detect_drone_mac.py --source video.mp4
```

## Datasets

Combined from multiple sources:
- Roboflow: Zhejiang University drones (4,231 images)
- Kaggle: [drone-dataset-uav](https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav)
- Kaggle: [yolo-drone-detection-dataset](https://www.kaggle.com/datasets/muki2003/yolo-drone-detection-dataset)

## OAK-1W Deployment

1. Train model using Kaggle notebook
2. Download `.blob` file
3. Copy to OAK device
4. Run `detect_drone_oak.py`

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for training)
- ONNX Runtime (for inference)
- OpenCV
- OAK DepthAI (for OAK deployment)