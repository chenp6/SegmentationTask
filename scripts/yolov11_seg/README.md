# YOLOv11 Segmentation Training with Roboflow

This module provides training scripts for YOLOv11 instance segmentation using the
same COCO dataset root as the Mask2Former pipeline.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the shared COCO dataset used by Mask2Former under `data/hospital_coco`.

## Configuration

Edit `config.py` to set:
- Shared dataset root (`data_root`)
- Model size (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
- Training parameters

## Training

### Train from the shared COCO dataset
```bash
python -m scripts.yolov11_seg.train
```

### Use a different shared COCO dataset root
```bash
python -m scripts.yolov11_seg.train --data-root path/to/hospital_coco
```

### Fine-tune from pretrained YOLOv11 segmentation weights
```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
model.train(data="data/hospital_coco/yolo/data.yaml", epochs=100, imgsz=640)
```

The training script will generate `data/hospital_coco/yolo/data.yaml` plus YOLO
segmentation labels from the same COCO annotations used by Mask2Former.

## Evaluation

```bash
python -m scripts.yolov11_seg.evaluate --model output/yolov11/exp/weights/best.pt --data data.yaml
```

## Visualization

```bash
python -m scripts.yolov11_seg.visualize --model output/yolov11/exp/weights/best.pt --source data/test/images --max-images 10
```

## Notes

- YOLOv11 uses YOLOv8 format for segmentation
- Make sure your Roboflow dataset is exported in YOLOv8 format
- Adjust class names in `dataset.py` based on your dataset
