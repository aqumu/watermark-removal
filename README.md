# Watermark Removal

A monorepo for training deep learning models to remove semi-transparent watermarks from images. The project has two components: a **synthetic dataset generator** and a **training pipeline** containing a removal model and a segmentation model.

## How It Works

1. **Pipeline** — downloads clean images, composites your watermark template onto them with randomized placement, opacity, and degradation (JPEG/WebP/noise), and saves each sample as a `(clean, watermarked, mask)` triplet.
2. **Segmentation training** — trains a small U-Net to predict the watermark mask from a watermarked image alone. This removes the need to supply a mask at inference time.
3. **Removal training** — trains a Masked U-Net on those triplets. The model takes a watermarked image + binary mask as input and predicts the residual to subtract, reconstructing the clean image.
4. **Inference** — loads both trained checkpoints, automatically predicts the mask with the segmentation model, then removes the watermark and outputs the clean result at the original resolution.

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
    ├── train.py                # Removal model training entry point
    ├── train_seg.py            # Segmentation model training entry point
    ├── infer.py                # Removal inference entry point (mask optional)
    ├── infer_seg.py            # Segmentation inference entry point (mask only)
    ├── configs/
    │   ├── train.yaml          # Removal model config
    │   └── seg.yaml            # Segmentation model config
    └── src/
        ├── model.py            # MaskedUNet architecture (removal)
        ├── seg_model.py        # SegUNet architecture (segmentation)
        ├── trainer.py          # Removal training loop, checkpointing, logging
        ├── seg_trainer.py      # Segmentation training loop (BCE+Dice, IoU)
        ├── dataset.py          # WatermarkDataset with mask augmentation
        ├── seg_dataset.py      # WatermarkSegDataset (watermarked → mask)
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

### 2. Train the Segmentation Model

```bash
cd training
python train_seg.py
```

Reads `configs/seg.yaml`. Trains a small U-Net to predict the watermark mask from a watermarked image alone. Checkpoints are saved to `checkpoints_seg/` and metrics logged to `runs_seg/`.

To resume an interrupted run:

```bash
python train_seg.py --resume checkpoints_seg/epoch_0010.pth
```

### 3. Train the Removal Model

```bash
cd training
python train.py
```

Reads `configs/train.yaml`. Checkpoints are saved to `checkpoints/` and metrics logged to `runs/train.csv` and `runs/val.csv`.

To resume an interrupted run:

```bash
python train.py --resume checkpoints/epoch_0010.pth
```

### 4. Inspect the Predicted Mask (optional)

```bash
cd training
python infer_seg.py \
  --checkpoint  checkpoints_seg/epoch_0030.pth \
  --watermarked input.jpg \
  --output      mask.png
```

Useful for verifying segmentation quality before running the full pipeline.

### 5. Run Inference

With automatic mask prediction (recommended):

```bash
cd training
python infer.py \
  --checkpoint     checkpoints/epoch_0060.pth \
  --seg-checkpoint checkpoints_seg/epoch_0030.pth \
  --watermarked    input.jpg \
  --output         result.png
```

With a manually provided mask:

```bash
python infer.py \
  --checkpoint  checkpoints/epoch_0060.pth \
  --watermarked input.jpg \
  --mask        mask.png \
  --output      result.png
```

To generate a normalized loss heatmap during debugging, provide the clean ground truth and the `--debug` flag:

```bash
python infer.py \
  --checkpoint  checkpoints/epoch_0060.pth \
  --watermarked input.jpg \
  --mask        mask.png \
  --output      result.png \
  --clean       clean.png \
  --debug
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

### Segmentation (`training/configs/seg.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.root` | `../pipeline/dataset` | Path to generated dataset |
| `dataset.image_size` | `256` | Training image size |
| `dataset.max_samples` | `200` | Cap samples (set to null for all) |
| `model.base_channels` | `16` | Feature width at first encoder stage |
| `model.depth` | `3` | Number of encoder/decoder stages |
| `training.epochs` | `30` | Training epochs |
| `training.lr` | `2e-4` | Learning rate (AdamW) |

### Removal (`training/configs/train.yaml`)

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

## Model Architectures

### SegUNet (segmentation)

Predicts the watermark mask from a watermarked image alone.

```
Input:   3 channels  (RGB watermarked image)
Output:  1 channel   (soft mask in [0, 1], thresholded at 0.5 for inference)

Encoder: 3 stages — channels 16 → 32 → 64, each Conv-BN-ReLU × 2 + MaxPool
Bridge:  128 channels
Decoder: 3 stages with bilinear upsampling + skip connections
Head:    1×1 Conv → Sigmoid
~500K parameters
```

Trained with soft alpha masks as targets (not binarized) so the model learns feathered edges accurately. The removal model's robustness to imperfect masks means even approximate segmentation output produces good results.

### MaskedUNet (removal)

Removes the watermark given the image and a binary mask.

```
Input:   4 channels  (RGB watermarked image + binary mask)
Output:  3 channels  (residual delta to subtract)

Encoder: 4 stages — channels 32 → 64 → 128 → 256, each Conv-BN-ReLU × 2 + MaxPool
Bridge:  512 channels
Decoder: 4 stages with bilinear upsampling + skip connections
Head:    1×1 Conv → Sigmoid
~7.85M parameters
```

**Prediction formula:**

```
clean = watermarked - model_output   (clamped to valid range)
```

The residual formulation is naturally correct outside the watermark region — the model outputs ~0 where there is no watermark, leaving clean pixels untouched.

---

## Loss Functions

### Segmentation loss (BCE + Dice)

| Component | Purpose |
|-----------|---------|
| BCE | Pixel-wise classification with soft targets |
| Dice | Overlap-based term; handles class imbalance at mask boundaries |

Validation metric: **IoU** at threshold 0.5.

### Removal loss

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

- **Two-stage pipeline** — a dedicated segmentation model predicts the mask; the removal model uses it. This separates the detection and restoration tasks cleanly.
- **Soft alpha targets for segmentation** — the segmentation model is trained on the raw soft alpha masks (not binarized) so it learns feathered edges as accurately as possible.
- **Mask augmentation for removal** — during removal training, masks are randomly binarized or blurred to build robustness to the imperfect masks the segmentation model will produce at inference.
- **Soft alpha masks** — pipeline masks encode the watermark's feathered edges (0–255), not just binary inside/outside. The removal model explicitly learns the transition zone.
- **Mixed blending modes** — 50% of samples are blended in linear RGB, 50% in sRGB, improving generalization.
- **Realistic degradation** — JPEG/WebP compression, downscaling, and Gaussian noise simulate real-world image handling.
- **Resume support** — the downloader tracks failed URLs and both training scripts support `--resume`.

---

## Requirements

- Python 3.10+
- PyTorch (installed by setup scripts)
- `opencv-python`, `Pillow`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `requests`
