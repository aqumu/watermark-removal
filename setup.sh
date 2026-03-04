#!/usr/bin/env bash
set -e

echo "=== Watermark Removal - Setup ==="
echo

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10-3.13."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip --quiet

echo
echo "Select your hardware:"
echo "  1) CPU only"
echo "  2) NVIDIA GPU - CUDA 12.1"
echo "  3) NVIDIA GPU - CUDA 11.8"
echo "  4) AMD GPU - ROCm 6.0 (Linux only)"
echo "  5) Skip PyTorch install (already installed)"
echo
read -rp "Enter choice [1-5]: " choice

case "$choice" in
    1) pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu ;;
    2) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 ;;
    3) pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 ;;
    4) pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0 ;;
    5) echo "Skipping PyTorch install." ;;
    *) echo "Invalid choice. Defaulting to CPU."
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu ;;
esac

echo
echo "Installing dependencies..."
pip install -r pipeline/requirements.txt
pip install -r training/requirements.txt
pip install -e pipeline/

echo
echo "=== Setup complete ==="
echo
echo "Generate the dataset (run from pipeline/ directory):"
echo "  cd pipeline && wm-generate && cd .."
echo
echo "Train the model:"
echo "  .venv/bin/python training/train.py"
echo
echo "Run inference:"
echo "  .venv/bin/python training/infer.py --checkpoint training/checkpoints/epoch_XXXX.pth --watermarked IMAGE --mask MASK --output result.png"
echo
