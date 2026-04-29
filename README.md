# Watermark Removal

A monorepo for training deep learning models to remove semi-transparent watermarks from images. The project has two main components: a **synthetic dataset generator** and a **training stack** containing a removal model and a segmentation model.

## How It Works

1. **Data generation** — downloads clean images, composites your watermark template onto them with randomized placement, opacity, and degradation (JPEG/WebP/noise), and saves each sample as a `(clean, watermarked, mask)` triplet.
2. **Segmentation training** — trains a small U-Net to predict the watermark mask from a watermarked image alone. This removes the need to supply a mask at inference time.
3. **Removal training** — trains a Masked U-Net on those triplets. The model takes a watermarked image + binary mask as input and predicts the residual to subtract, reconstructing the clean image.
4. **Inference** — loads both trained checkpoints, automatically predicts the mask with the segmentation model, then removes the watermark and outputs the clean result at the original resolution.

---

## Dashboard-First Workflow

The recommended way to operate the training stack is through the dashboard shell:

```bash
python start.py
```

This starts the dashboard server against `training/runs/`, loads the latest run history if one exists, and gives you a modal-first workspace where you can:

- inspect an existing run without starting training
- create a draft config
- start a fresh run
- start from a compatible checkpoint
- pause and resume the active job

For a standalone inspect-only dashboard, use:

```bash
python training/serve_dashboard.py --family-dir training/runs
```

For the compact operator workflow reference, see [docs/README.md](docs/README.md).

Direct CLI training is still available for automation and experiments that do not need the dashboard shell.

## Repository Structure

```text
watermark-removal/
  data_gen/                   # Synthetic dataset generation
    configs/default.yaml
    watermark.png
    watermark_gen/
  configs/
    profiles/
  wm_shared/                  # Shared preprocessing + profile/config loading
  training/
    artifacts/
      checkpoints/
    runs/
    scripts/
      rebuild_preprocessed_store.py
      cleanup_legacy_dataset_sidecars.py
    train.py
    train_seg.py
    infer.py
    infer_seg.py
    configs/
    src/
      common/                 # Plotting, checkpoints, metrics
      tasks/
        removal/              # Removal dataset, model, losses, trainer
        segmentation/         # Segmentation dataset, model, trainer
  setup.sh
  setup.bat
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
cd data_gen
pip install -e .
wm-generate
```

This reads `configs/default.yaml`, downloads up to 15,000 images from `data/sql.csv`, and produces:

```
data_gen/dataset/
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

### 2. Start the Dashboard

```bash
python start.py
```

Use the launcher to inspect existing runs or create a new draft workspace.

### 3. Train the Segmentation Model

```bash
cd training
python train_seg.py
```

Reads `configs/seg.yaml`. Trains a small U-Net to predict the watermark mask from a watermarked image alone. Checkpoints are saved to `artifacts/checkpoints/segmentation/` and metrics logged to `runs/segmentation/`.

The legacy live plotter window opens during training. The run folder also stores tracker-neutral metadata under `meta/` and preview images under `previews/` so a future experiment manager can ingest them cleanly.

To resume an interrupted run:

```bash
python train_seg.py --resume artifacts/checkpoints/segmentation/epoch_0010.pth
```

### 4. Train the Removal Model

```bash
cd training
python train.py
```

Reads `configs/train_512.yaml`. Checkpoints are saved to `artifacts/checkpoints/removal_512/` and metrics logged to `runs/removal_512/`.

Removal training can use a mirrored deterministic preprocessing store at
`data_gen/preprocessed_store/`. It is keyed by the current removal pipeline
settings, so changing ROI dimensions or crop settings moves training onto a new
namespace instead of reusing stale artifacts from older preprocessing logic.

To resume an interrupted run:

```bash
python train.py --resume artifacts/checkpoints/removal/epoch_0010.pth
```

To clear the current removal preprocessing store namespace before training:

```bash
python train.py --clear-preprocessed-store
```

To clear and fully regenerate it before training:

```bash
python train.py --rebuild-preprocessed-store
```

Standalone maintenance entrypoints live under `training/scripts/`:

```bash
python training/scripts/rebuild_preprocessed_store.py --clear-first
python training/scripts/cleanup_legacy_dataset_sidecars.py --dry-run
```

The legacy live plotter window opens during training. The run folder also stores tracker-neutral metadata under `meta/`, preview images under `previews/`, and artifact references under `artifacts/`.

### 5. Inspect the Predicted Mask (optional)

```bash
cd training
  python infer_seg.py \
  --checkpoint  artifacts/checkpoints/segmentation/epoch_0030.pth \
  --watermarked input.jpg \
  --output      mask.png
```

Useful for verifying segmentation quality before running the full workflow.

### 6. Run Inference

With automatic mask prediction (recommended):

```bash
cd training
  python infer.py \
  --checkpoint     artifacts/checkpoints/removal/epoch_0060.pth \
  --seg-checkpoint artifacts/checkpoints/segmentation/epoch_0030.pth \
  --watermarked    input.jpg \
  --output         result.png
```

With a manually provided mask:

```bash
  python infer.py \
  --checkpoint  artifacts/checkpoints/removal/epoch_0060.pth \
  --watermarked input.jpg \
  --mask        mask.png \
  --output      result.png
```

To generate a normalized loss heatmap during debugging, provide the clean ground truth and the `--debug` flag:

```bash
  python infer.py \
  --checkpoint  artifacts/checkpoints/removal/epoch_0060.pth \
  --watermarked input.jpg \
  --mask        mask.png \
  --output      result.png \
  --clean       clean.png \
  --debug
```

---

## Configuration

### Data Generation (`data_gen/configs/default.yaml`)

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
| `dataset.root` | `../data_gen/dataset` | Path to generated dataset |
| `dataset.image_size` | `256` | Training image size |
| `dataset.max_samples` | `10000` | Cap samples (set to null for all) |
| `model.encoder` | `efficientnet-b0` | Pretrained encoder used by the segmentation U-Net |
| `training.epochs` | `40` | Training epochs |
| `training.lr` | `1e-4` | Decoder / head learning rate |

### Removal (`training/configs/train_512.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.root` | `../data_gen/dataset` | Path to generated dataset |
| `dataset.image_width` | `512` | Removal ROI width |
| `dataset.image_height` | `256` | Removal ROI height |
| `dataset.max_samples` | `15000` | Cap samples (set to null for all) |
| `training.epochs` | `400` | Training epochs |
| `training.batch_size` | `8` | Batch size |
| `training.lr` | `1.5e-4` | Learning rate (AdamW) |
| `training.lr_scheduler` | `cosine_warmup` | LR schedule |

---

## Model Architectures

### Segmentation Model

Predicts the watermark mask from a watermarked image alone.

```
Input:   3 channels  (RGB watermarked image)
Output:  1 channel   (soft mask in [0, 1], thresholded at 0.5 for inference)

Encoder: pretrained EfficientNet-B0
Decoder: U-Net style decoder from segmentation-models-pytorch
Head:    1×1 convolution producing soft mask logits
```

Trained with soft alpha masks as targets (not binarized) so the model learns feathered edges accurately. The removal model's robustness to imperfect masks means even approximate segmentation output produces good results.

## Run Artifacts

Each training run writes into a stable artifact layout under `training/runs/<experiment>/`:

- `train.csv` / `val.csv` for scalar history
- `meta/` for resolved config and model overview
- `previews/` for epoch-by-epoch inference snapshots
- `artifacts/` for tracker-ready artifact references
- `tracker/` reserved for a future experiment-platform adapter

### Removal Model

Removes the watermark given the image, a mask hint, and a gradient cue.

```
Input:   5 channels  (RGB watermarked image + mask hint + grayscale gradient)
Output:  3 channels  (residual delta to subtract)

Encoder: 4 stages — channels 32 → 64 → 128 → 256, each Conv-BN-ReLU × 2 + MaxPool
Bridge:  512 channels
Decoder: nearest-neighbor upsampling + skip connections
Head:    1×1 convolution producing residual delta
```

**Prediction formula:**

```
clean = watermarked - model_output   (clamped to valid range)
```

The residual formulation is naturally correct outside the watermark region — the model outputs ~0 where there is no watermark, leaving clean pixels untouched.

---

## Loss Functions

### Segmentation loss

| Component | Purpose |
|-----------|---------|
| BCE | Pixel-wise classification with soft targets |
| Focal | Extra emphasis on hard pixels and weak watermark regions |
| L1 | Direct soft-mask regression |
| Multi-scale L1 | Penalizes hollow/outline-only predictions |
| Dice | Overlap-based term; handles class imbalance at mask boundaries |

Validation metric: **IoU** at threshold 0.5.

### Removal loss

| Component | Purpose |
|-----------|---------|
| Masked L1 | Pixel accuracy inside the watermark interior |
| Border ring | Penalizes residual edges in the transition zone |
| Perceptual | VGG16 feature matching for texture fidelity |
| Color moment | Keeps restored region brightness / tint stable |
| Saturation | Recovers desaturated colors under the white watermark |
| Background TV | Suppresses structured artifacts outside the watermark |
| Background delta | Forces near-zero edits outside the watermark |
| Edge coherence | Penalizes coherent leftover letter fragments |

Perceptual loss frequency and all term weights are configurable.

---

## Key Design Decisions

- **Two-stage workflow** — a dedicated segmentation model predicts the mask; the removal model uses it. This separates the detection and restoration tasks cleanly.
- **Soft alpha targets for segmentation** — the segmentation model is trained on the raw soft alpha masks (not binarized) so it learns feathered edges as accurately as possible.
- **Mask-guided ROI cropping for removal** — removal training and inference crop a fixed-aspect watermark region before resizing to square, keeping watermark geometry consistent across source aspect ratios.
- **Mask augmentation for removal** — the removal model still sees imperfect mask hints during training so it stays robust to small mask errors at inference.
- **Soft alpha masks** — dataset masks encode the watermark's feathered edges (0–255), not just binary inside/outside. The removal model explicitly learns the transition zone.
- **Mixed blending modes** — 50% of samples are blended in linear RGB, 50% in sRGB, improving generalization.
- **Realistic degradation** — JPEG/WebP compression, downscaling, and Gaussian noise simulate real-world image handling.
- **Resume support** — the downloader tracks failed URLs and both training scripts support `--resume`.

---

## Requirements

- Python 3.10+
- PyTorch (installed by setup scripts)
- `opencv-python`, `Pillow`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `requests`
