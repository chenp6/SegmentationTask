"""
Configuration for the SAM3 instance segmentation pipeline.

Supports both UNet (center heatmap) and query-based (Mask2Former-lite) decoders.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    data_root: str = "data/hospital_coco"
    image_size: int = 1024  # SAM3 native resolution


@dataclass
class ModelConfig:
    # SAM3 backbone
    sam3_model_name: str = "facebook/sam3"
    sam3_checkpoint: str = None
    sam3_config: str = None

    # Adapter settings
    use_adapters: bool = True
    adapter_dim: int = 64

    # Number of classes (set automatically from dataset)
    num_classes: int = 26

    # Decoder type: "query" (Mask2Former-lite) or "unet" (center heatmap)
    decoder_type: str = "query"

    # Query decoder settings
    num_queries: int = 30
    decoder_layers: int = 3
    d_model: int = 256
    nhead: int = 8
    dim_ff: int = 1024

    # Output directory
    output_dir: str = "output/sam3"


@dataclass
class TrainConfig:
    # Batch configuration
    batch_size: int = 2
    grad_accum_steps: int = 4
    num_epochs: int = 40
    num_workers: int = 0

    # Optimizer
    learning_rate: float = 1e-4
    adapter_lr_factor: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LR schedule
    lr_warmup_steps: int = 200

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Loss weights (query-based)
    cls_weight: float = 2.0
    mask_bce_weight: float = 5.0
    mask_dice_weight: float = 5.0
    no_object_weight: float = 0.1   # down-weight "no object" class in CE

    # Logging & checkpointing
    log_every: int = 10
    save_every_epochs: int = 5


@dataclass
class AugConfig:
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.0
    rotate_limit: int = 15
    brightness: float = 0.2
    contrast: float = 0.2
