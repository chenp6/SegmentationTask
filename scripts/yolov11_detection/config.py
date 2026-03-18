"""
Configuration for YOLOv11 object detection training.

This module targets the original Roboflow COCO detection dataset used by
`scripts.object_detection_to_image_segmentaion`.
"""
from dataclasses import dataclass

import torch


@dataclass
class DataConfig:
    # Original object detection COCO dataset root.
    data_root: str = "data/hiod_coco"
    # Local YOLO detection view derived from the COCO annotations above.
    yolo_dataset_dir: str = "data/hiod_coco/yolo_detection"
    # Ultralytics-compatible YAML generated under the YOLO dataset view.
    data_yaml_path: str = "data/hiod_coco/yolo_detection/data.yaml"
    # Default inference / training resolution for YOLOv11 detection.
    image_size: int = 640


@dataclass
class ModelConfig:
    # Detection checkpoint to fine-tune from.
    model_name: str = "yolo11n.pt"
    pretrained: bool = True
    # Architecture YAML used when training from scratch.
    scratch_model_cfg: str = "yolo11n.yaml"

    # Training parameters.
    epochs: int = 100
    batch_size: int = 16
    patience: int = 50

    # Output directory for Ultralytics runs.
    output_dir: str = "output/yolov11_detection"


@dataclass
class TrainConfig:
    # Auto-select GPU when available.
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 0

    # Optimization.
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Augmentation.
    hsv_h: float = 0
    hsv_s: float = 0
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.0
    scale: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.0
    mixup: float = 0.0

    # Logging.
    save_period: int = 10
    project: str = "yolov11_detection"
    name: str = "exp"

    # Validation.
    val: bool = True
    split: str = "val"
