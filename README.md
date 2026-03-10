# Hospital Object Segmentation Pipeline

Instance segmentation for **26 hospital object classes** (C-Arm, Patient bed, Monitor, Saline stand, etc.) using a COCO-format dataset from Roboflow.

Two architectures are implemented:

| Pipeline | Architecture | Run command |
|----------|-------------|-------------|
| **Mask2Former** | Swin-Large + transformer decoder | `python -m scripts.mask2former_seg.train` |
| **SAM2-UNet** | SAM2 Hiera-Large backbone + adapters + UNet decoder | `python -m scripts.sam2_seg.train` |

Both evaluate using **COCO mAP50** via `pycocotools`.

---

## Prerequisites

- **NVIDIA DGX Spark** (or any GPU with ≥16GB VRAM)
- Docker with NVIDIA Container Toolkit
- Dataset at `data/hospital_coco/` (train/valid/test splits with `_annotations.coco.json`)

## Quick Start (Docker)

### What To Do

If you just changed the Dockerfile and want to fix `ModuleNotFoundError: No module named 'ultralytics'`, run these commands in order:

```bash
# 1. Remove the old container if it exists
docker rm -f seg_pipeline

# 2. Rebuild the image so the new dependencies are installed
docker build -t segmentation_pipeline:latest .

# 3. Start a new container
docker run -d \
    --name seg_pipeline \
    --gpus all \
    --shm-size=8g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e HF_HOME=/workspace/.cache/huggingface \
    -v "$(pwd)/scripts:/workspace/scripts" \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/hf_cache:/workspace/.cache/huggingface" \
    -w /workspace \
    segmentation_pipeline:latest \
    tail -f /dev/null

# 4. Confirm the container is running
docker ps

# 5. Confirm ultralytics is installed inside the container
docker exec seg_pipeline python -c "from ultralytics import YOLO; print('ultralytics ok')"

# 6. Start YOLOv11 training
docker exec seg_pipeline python -m scripts.yolov11_seg.train
```

If step 5 fails, you are still using an old image or old container.
If YOLOv11 fails with `unable to allocate shared memory(shm)`, keep `--shm-size=8g`
and run with `workers=0` or `--workers 0`.

### 1. Build the Docker image

```bash
docker build -t segmentation_pipeline:latest .
```

### 2. Start the container

```bash
docker run -d \
    --name seg_pipeline \
    --gpus all \
    --shm-size=8g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e HF_HOME=/workspace/.cache/huggingface \
    -v "$(pwd)/scripts:/workspace/scripts" \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/hf_cache:/workspace/.cache/huggingface" \
    -w /workspace \
    segmentation_pipeline:latest \
    tail -f /dev/null
```

### 2a. Confirm the container is running

```bash
docker ps
docker exec seg_pipeline python -c "from ultralytics import YOLO; print('ultralytics ok')"
```

### 2b. If YOLOv11 hits shared memory errors

This error:

```text
RuntimeError: unable to allocate shared memory(shm)
```

usually means Docker shared memory is too small for multi-worker dataloading.

Use both of these:

```bash
# Start container with larger shared memory
docker run -d \
    --name seg_pipeline \
    --gpus all \
    --shm-size=8g \
    ...

# Run YOLOv11 with safe dataloader worker count
docker exec seg_pipeline python -m scripts.yolov11_seg.train --workers 0
```

The project default for YOLOv11 is now `workers=0` in Docker-safe mode.

### 3. Run a pipeline

```bash
# Mask2Former
docker exec seg_pipeline python -m scripts.mask2former_seg.train
docker exec seg_pipeline python -m scripts.mask2former_seg.evaluate
docker exec seg_pipeline python -m scripts.mask2former_seg.visualize

# SAM2-UNet
docker exec seg_pipeline python -m scripts.sam2_seg.train
docker exec seg_pipeline python -m scripts.sam2_seg.evaluate
docker exec seg_pipeline python -m scripts.sam2_seg.visualize
```

---

## Pipeline 1: Mask2Former

**Architecture:** [Mask2Former](https://arxiv.org/abs/2112.01527) with a Swin-Large backbone pretrained on COCO instance segmentation.

### Files

| File | Purpose |
|------|---------|
| `scripts/mask2former_seg/config.py` | Dataclass configs (model, training, augmentation) |
| `scripts/mask2former_seg/dataset.py` | COCO dataset → Mask2Former processor format |
| `scripts/mask2former_seg/train.py` | Training loop with HuggingFace Mask2Former |
| `scripts/mask2former_seg/evaluate.py` | COCO mAP50 evaluation |
| `scripts/mask2former_seg/visualize.py` | Overlay predictions on test images |
| `scripts/mask2former_seg/download_dataset.py` | Download dataset from Roboflow |

### Training

```bash
# Default: 40 epochs, batch_size=2, grad_accum=4
docker exec seg_pipeline python -m scripts.mask2former_seg.train

# Custom
docker exec seg_pipeline python -m scripts.mask2former_seg.train \
    --epochs 80 --batch_size 2

# Resume from checkpoint
docker exec seg_pipeline python -m scripts.mask2former_seg.train \
    --resume output/mask2former/checkpoint_epoch20
```

### Evaluation

```bash
docker exec seg_pipeline python -m scripts.mask2former_seg.evaluate \
    --checkpoint output/mask2former/best_model \
    --split test
```

### Key Settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| Checkpoint | `facebook/mask2former-swin-large-coco-instance` | COCO pretrained |
| Image size | 512 | Processor handles resizing |
| LR (head) | 1e-5 | Higher than backbone |
| LR (backbone) | 1e-6 | 0.1× head LR |
| Precision | bfloat16 | Requires Ampere+ GPU |
| Grad clip | 0.01 | Very conservative |

---

## Pipeline 2: SAM2-UNet

**Architecture:** SAM2's Hiera-Large image encoder (frozen) with lightweight 1×1 conv adapters and a UNet-style decoder for semantic segmentation. Instance predictions are extracted via connected-component post-processing.

```
Input Image → [SAM2 Hiera Encoder (frozen)] → [Adapters] → [UNet Decoder] → Semantic Mask → Connected Components → Instance Predictions
```

### Files

| File | Purpose |
|------|---------|
| `scripts/sam2_seg/config.py` | Dataclass configs (SAM2 model, adapters, training) |
| `scripts/sam2_seg/dataset.py` | COCO dataset → normalised tensors + semantic masks |
| `scripts/sam2_seg/sam2_backbone.py` | SAM2 Hiera encoder as multi-scale feature extractor |
| `scripts/sam2_seg/adapters.py` | 1×1 conv bottleneck adapters (parameter-efficient) |
| `scripts/sam2_seg/unet_decoder.py` | UNet decoder with skip connections |
| `scripts/sam2_seg/segmentation_model.py` | Full model: backbone + adapters + decoder |
| `scripts/sam2_seg/train.py` | Training loop (CE + Dice loss) |
| `scripts/sam2_seg/evaluate.py` | COCO mAP50 evaluation (semantic → instance conversion) |
| `scripts/sam2_seg/visualize.py` | Overlay predictions on test images |

### Training

```bash
# Default: 40 epochs, batch_size=2, grad_accum=4
docker exec seg_pipeline python -m scripts.sam2_seg.train

# Custom
docker exec seg_pipeline python -m scripts.sam2_seg.train \
    --epochs 80 --batch_size 1 --lr 5e-4

# Resume
docker exec seg_pipeline python -m scripts.sam2_seg.train \
    --resume output/sam2/checkpoint_epoch10
```

### Evaluation

```bash
docker exec seg_pipeline python -m scripts.sam2_seg.evaluate \
    --checkpoint output/sam2/best_model \
    --split test
```

### Visualization

```bash
docker exec seg_pipeline python -m scripts.sam2_seg.visualize \
    --split test --max_images 50
```

### Key Settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| SAM2 model | `facebook/sam2.1-hiera-large` | Auto-downloaded from HuggingFace |
| Image size | 1024 | SAM2 native resolution |
| LR (decoder) | 1e-4 | Main learning rate |
| LR (adapters) | 1e-5 | 0.1× decoder LR |
| Loss | CE + Dice | Equal weighting |
| Trainable params | ~4.2M | Out of 216M total (adapters + decoder only) |
| Precision | bfloat16 | Requires Ampere+ GPU |

---

## Dataset

The hospital COCO dataset is stored at `data/hospital_coco/` with the following structure:

```
data/hospital_coco/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg  (17,250 images)
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg  (1,008 images)
└── test/
    ├── _annotations.coco.json
    └── *.jpg  (521 images)
```

**26 classes:** C-Arm, Anesthesia machine, Bed-side-table, Bin, C-Arm Monitor, Chair, Door, Door Handler, Foot stool, Laparoscopy, Ligasure machine, Machine, Medicine trolley, Monitor, Oxygen Tank, Patient bed, Patient table, Pillow, Rack, Saline stand, Sofa, Stand, Stool, Theatre suction trolley, Torriniqut machine.

To re-download:

```bash
docker exec seg_pipeline python -m scripts.mask2former_seg.download_dataset
```

Requires `roboflow_credentials.json` in the project root.

---

## Output Structure

```
output/
├── mask2former/
│   ├── best_model/           # Best validation loss checkpoint
│   ├── checkpoint_epoch5/    # Periodic snapshots
│   └── visualizations/test/  # Overlay images
└── sam2/
    ├── best_model/model.pt   # Best validation loss checkpoint
    ├── checkpoint_epoch5/    # Periodic snapshots
    └── visualizations/test/  # Overlay images
```

---

## Research

See [docs/segmentation_research.md](docs/segmentation_research.md) for a detailed analysis comparing all approaches, pros/cons, and future improvement suggestions.
