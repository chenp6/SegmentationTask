"""
Configuration for YOLOv11 instance segmentation training.

Uses the same COCO dataset root as Mask2Former/SAM pipelines and derives a
YOLO-seg dataset view locally.
"""
import torch
from dataclasses import dataclass


@dataclass
class DataConfig:
    # Same source dataset as mask2former/sam pipelines
    data_root: str = "data/hospital_coco"
    yolo_dataset_dir: str = "data/hospital_coco/yolo"
    data_yaml_path: str = "data/hospital_coco/yolo/data.yaml"
    image_size: int = 640  # YOLOv11 default


@dataclass
class ModelConfig:
    # YOLOv11 model
    model_name: str = "yolo11n-seg.pt"  # nano version for segmentation
    pretrained: bool = True
    scratch_model_cfg: str = "yolo11n-seg.yaml"

    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    patience: int = 50  # early stopping

    # Output directory
    output_dir: str = "output/yolov11"


@dataclass
class TrainConfig:
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    workers: int = 0

    # Optimization
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0

    # Logging
    save_period: int = 10
    project: str = "yolov11_seg"
    name: str = "exp"

    # Validation
    val: bool = True
    split: str = "val"
