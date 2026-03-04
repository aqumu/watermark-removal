# Watermark Removal

A monorepo for training a deep learning model to remove semi-transparent watermarks from images. The project has two components: a **synthetic dataset generator** and a **U-Net training pipeline**.

## How It Works

1. **Pipeline** — downloads clean images, composites your watermark template onto them with randomized placement, opacity, and degradation (JPEG/WebP/noise), and saves each sample as a `(clean, watermarked, mask)` triplet.
2. **Training** — trains a Masked U-Net on those triplets. The model takes a watermarked image + soft mask as input and predicts the residual to subtract, reconstructing the clean image.
3. **Inference** — loads a trained checkpoint, processes an image+mask pair, and outputs the clean result at the original resolution.

---

## Repository Structure

```
watermark-removal/
├── setup.sh / setup.bat        # Environment setup (interactive GPU selection)
│
├── pipeline/                   # Synthetic dataset generation
│   ├── watermark.png           # RGBA watermark template
│   ├── configs/default.yaml    # Generation config
│   ├── data/sql.csv            # Source image URLs
│   └── watermark_gen/
│       ├── cli/
│       │   ├── generate.py     # Entry point: wm-generate
│       │   └── preview.py      # Entry point: wm-preview
│       └── core/
│           ├── downloader.py   # Parallel image downloader
│           ├── dataset.py      # Multiprocess sample generation
│           ├── watermark.py    # Alpha blending (linear/sRGB)
│           ├── degrade.py      # JPEG/WebP/noise degradation
│           └── io.py           # Sample serialization
│
└── training/                   # Model training and inference
    ├── train.py                # Training entry point
    ├── infer.py                # Inference entry point
    ├── configs/train.yaml      # Training config
    └── src/
        ├── model.py            # MaskedUNet architecture
        ├── trainer.py          # Training loop, checkpointing, logging
        ├── dataset.py          # WatermarkDataset with mask augmentation
        └── losses.py           # L1, SSIM, Perceptual, BorderRing losses
```

---

## Setup

```bash
# Linux / macOS
./setup.sh

# Windows
setup.bat
```

Both scripts prompt you to select a backend (CUDA 12.1, CUDA 11.8, AMD ROCm, or CPU), create a virtual environment, and install PyTorch and all dependencies.

---

## Usage

### 1. Generate the Dataset

```bash
cd pipeline
pip install -e .
wm-generate
```

This reads `configs/default.yaml`, downloads up to 15,000 images from `data/sql.csv`, and produces:

```
pipeline/dataset/
├── sample_00000/
│   ├── clean.png
│   ├── watermarked.jpg
│   ├── mask.png
│   └── meta.json
...
```

To inspect a random sample before generating the full dataset:

```bash
wm-preview
```

### 2. Train the Model

```bash
cd training
python train.py
```

Reads `configs/train.yaml`. Checkpoints are saved to `checkpoints/` and metrics logged to `runs/train.csv` and `runs/val.csv`.

### 3. Run Inference

```bash
cd training
python infer.py \
  --checkpoint checkpoints/epoch_0060.pth \
  --watermarked input.jpg \
  --mask mask.png \
  --output result.png
```

---

## Configuration

### Pipeline (`pipeline/configs/default.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `download.num_images` | `15000` | Number of images to download |
| `download.workers` | `8` | Parallel download threads |
| `placement.width_fraction` | `[0.76, 0.82]` | Watermark width as fraction of image width |
| `alpha.base_range` | `[0.1, 0.2]` | Watermark opacity range |
| `degradation.jpeg_quality` | `[80, 95]` | JPEG compression quality range |

### Training (`training/configs/train.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.root` | `../pipeline/dataset` | Path to generated dataset |
| `dataset.image_size` | `128` | Training crop size |
| `dataset.max_samples` | `3000` | Cap samples (set to null for all) |
| `training.epochs` | `60` | Training epochs |
| `training.batch_size` | `16` | Batch size |
| `training.lr` | `2e-4` | Learning rate (AdamW) |
| `training.lr_scheduler` | `cosine` | `cosine`, `step`, or `none` |

---

## Model Architecture

**MaskedUNet** — a standard encoder-decoder U-Net with skip connections, modified for 4-channel input.

```
Input:   4 channels  (RGB watermarked image + binary mask)
Output:  3 channels  (residual delta to subtract)

Encoder: 4 stages — channels 32 → 64 → 128 → 256, each Conv-BN-ReLU × 2 + MaxPool
Bridge:  512 channels
Decoder: 4 stages with bilinear upsampling + skip connections
Head:    1×1 Conv → Sigmoid
```

**Prediction formula:**

```
clean = watermarked - model_output   (clamped to valid range)
```

The residual formulation is naturally correct outside the watermark region — the model outputs ~0 where there is no watermark, leaving clean pixels untouched.

---

## Loss Function

The training loss combines six terms:

| Component | Weight | Purpose |
|-----------|--------|---------|
| L1 Full | 1.0 | Pixel accuracy across the full image |
| L1 Masked | 4.0 | Extra focus on watermark region |
| SSIM | 1.0 | Structural consistency |
| Perceptual | 0.1 | VGG16 feature matching (texture) |
| Color Moment | 2.0 | Per-channel mean in mask region |
| Border Ring | 1.5 | Penalizes artifacts at feathered edges |

Perceptual loss is computed every 4 steps to reduce overhead.

---

## Key Design Decisions

- **Soft alpha masks** — masks encode the watermark's feathered edges (0–255), not just binary inside/outside. The model explicitly learns the transition zone.
- **Mask augmentation** — during training, masks are randomly binarized or blurred to build robustness to imperfect masks at inference time.
- **Mixed blending modes** — 50% of samples are blended in linear RGB, 50% in sRGB, improving generalization.
- **Realistic degradation** — JPEG/WebP compression, downscaling, and Gaussian noise simulate real-world image handling.
- **Resume support** — the downloader tracks failed URLs and the dataset generator can resume interrupted runs.

---

## Requirements

- Python 3.10+
- PyTorch (installed by setup scripts)
- `opencv-python`, `Pillow`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `requests`
