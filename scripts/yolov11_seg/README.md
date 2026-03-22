# YOLOv11 Segmentation Training with Roboflow

This module provides training scripts for YOLOv11 instance segmentation using the
same COCO dataset root as the Mask2Former pipeline.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the shared COCO dataset used by Mask2Former under `data/hospital_coco`.

### Colab

If you want to run this on Google Colab instead of Docker, the simplest flow is:

```bash
git clone <your-repo-url>
cd SegmentationTask
pip install -r scripts/yolov11_seg/requirements.txt
python -m scripts.mask2former_seg.download_dataset \
  --output_dir /content/data/hospital_coco \
  --credentials /content/roboflow_credentials.json
python -m scripts.yolov11_seg.prepare_dataset \
  --data-root /content/data/hospital_coco
python -m scripts.yolov11_seg.train \
  --data-yaml /content/data/hospital_coco/yolo/data.yaml \
  --workers 2
```

You can also skip the credentials JSON and pass Roboflow values directly:

```bash
python -m scripts.mask2former_seg.download_dataset \
  --output_dir /content/data/hospital_coco \
  --api-key "$ROBOFLOW_API_KEY" \
  --workspace "<workspace>" \
  --project "<project>" \
  --version "<version>"
```

## Configuration

Edit `config.py` to set:
- Shared dataset root (`data_root`)
- Model size (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
- Training parameters

## Training

### 1. Prepare YOLO data.yaml and labels

```bash
python -m scripts.yolov11_seg.prepare_dataset
```

### Use a different shared COCO dataset root

```bash
python -m scripts.yolov11_seg.prepare_dataset --data-root path/to/hospital_coco
```

For Colab, a typical path is `/content/data/hospital_coco`.

This step generates `data/hospital_coco/yolo/data.yaml` plus YOLO segmentation
labels from the same COCO annotations used by Mask2Former.

### 2. Train from the prepared YAML
```bash
python -m scripts.yolov11_seg.train --data-yaml data/hospital_coco/yolo/data.yaml
```

### Save directly into a chosen experiment directory

```bash
python -m scripts.yolov11_seg.train \
  --data-yaml data/hospital_coco/yolo/data.yaml \
  --output-dir output/yolov11_seg/exp1
```

### Or still train directly from the shared COCO dataset
```bash
python -m scripts.yolov11_seg.train --data-root path/to/hospital_coco
```

### Fine-tune from pretrained YOLOv11 segmentation weights
```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
model.train(data="data/hospital_coco/yolo/data.yaml", epochs=100, imgsz=640)
```

## Evaluation

```bash
python -m scripts.yolov11_seg.evaluate \
  --model output/yolov11_seg/exp1/weights/best.pt \
  --data data/hospital_coco/yolo/data.yaml
```

## Visualization

```bash
python -m scripts.yolov11_seg.visualize --model output/yolov11_seg/exp1/weights/best.pt --source data/test/images --max-images 10
```

## Notes

- YOLOv11 uses YOLOv8 format for segmentation
- This repo converts the Roboflow COCO export into YOLO labels automatically
- In Colab, increase `--workers` from `0` if the runtime is stable enough
