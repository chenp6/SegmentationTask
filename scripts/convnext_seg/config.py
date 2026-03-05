"""Configuration for ConvNeXt + Prototype instance segmentation."""
from dataclasses import dataclass


@dataclass
class DataConfig:
    data_root: str = "data/hospital_coco"
    image_size: int = 512


@dataclass
class ModelConfig:
    # Backbone
    backbone: str = "convnext_small"   # timm model name
    d_model: int = 256                  # FPN output channels
    num_prototypes: int = 32            # prototype masks

    # Output
    output_dir: str = "output/convnext"
    num_classes: int = 27  # set at runtime


@dataclass
class TrainConfig:
    num_epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 2
    lr_warmup_steps: int = 500
    save_every_epochs: int = 5

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Loss weights
    cls_weight: float = 1.0
    bbox_weight: float = 2.0
    mask_weight: float = 1.0

    # Detection
    score_thresh: float = 0.05   # training NMS
    nms_thresh: float = 0.5
    max_detections: int = 100

    # Backbone LR
    backbone_lr_factor: float = 0.0  # 0 = frozen


@dataclass
class AugConfig:
    pass
