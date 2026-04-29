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
pip install -r data_gen/requirements.txt
pip install -r training/requirements.txt
pip install -e data_gen/

echo
echo "=== Setup complete ==="
echo
echo "--------------------------------------------------------------"
echo "IMPORTANT: to keep the venv active in your current shell, run:"
echo "  source setup.sh"
echo "instead of ./setup.sh. Or activate manually at any time with:"
echo "  source .venv/bin/activate"
echo "--------------------------------------------------------------"
echo
echo "First-time workflow on a new machine:"
echo
echo "  1) Generate the dataset:"
echo "       wm-generate --config data_gen/configs/default.yaml"
echo
echo "  2) Train the model:"
echo "       cd training"
echo "       python train.py"
echo
echo "  3) Run inference:"
echo "       python infer.py --checkpoint artifacts/checkpoints/removal/best.pth \\"
echo "           --watermarked IMAGE --mask MASK --output result.png"
echo
