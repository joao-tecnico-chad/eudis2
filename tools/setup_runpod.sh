#!/bin/bash
# Run this ON the RunPod instance after SSH-ing in.
# It sets up everything needed for training.

set -e

echo "=== RunPod Training Setup ==="

# 1. Extract dataset
echo "Extracting dataset..."
mkdir -p ~/dataset
cd ~/dataset
tar xzf ~/drone_dataset.tar.gz
echo "Dataset: $(ls images/train | wc -l) train, $(ls images/val | wc -l) val"

# 2. Install dependencies
echo "Installing dependencies..."
pip install roboflow blobconverter onnx onnxsim -q

# 3. Clone project
cd ~
if [ ! -d eudis2 ]; then
    git clone https://github.com/joao-tecnico-chad/eudis2.git
fi

# 4. Run training
cd ~/eudis2
echo ""
echo "=== Ready to train ==="
echo "Run: python tools/train_runpod.py"
echo "  --epochs 150  (default)"
echo "  --batch 64    (RTX 5090 32GB)"
echo ""
echo "Or test with: python tools/train_runpod.py --epochs 10"
