"""
Train YOLOv11 segmentation model using Ultralytics on the shared COCO dataset.

Usage:
python -m scripts.yolov11_seg.train  \
  --data-root data/medbin_dataset  \
  --epochs 120  \
  --batch-size 16  \
  --imgsz 960  \
  --hsv-h 0.01  \
  --hsv-s 0.2  \
  --hsv-v 0.15  \
  --degrees 5  \
  --translate 0.08  \
  --scale 0.15  \
  --shear 2  \
  --perspective 0.0  \
  --flipud 0.0  \
  --fliplr 0.5  \
  --mosaic 0.2  \
  --mixup 0.0
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import DataConfig, ModelConfig, TrainConfig
from .dataset import build_yolo_dataset_from_coco


def resolve_ultralytics_save_args(output_dir: str) -> tuple[str, str]:
    # Treat output_dir as the final experiment directory instead of a project root.
    resolved_dir = Path(output_dir).resolve()
    return str(resolved_dir.parent), resolved_dir.name


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on the shared COCO dataset")
    parser.add_argument("--data-yaml", type=str, default=None, help="Path to an existing YOLO data.yaml")
    parser.add_argument("--data-root", type=str, default=None, help="COCO dataset root shared with mask2former, e.g. data/hospital_coco")
    parser.add_argument("--model", type=str, default=None, help="Model name/size (overrides config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save trained model outputs")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image size for train/val/test resize")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers for Ultralytics; use 0 in Docker if shm is limited")
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation strength")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation strength")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation strength")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation augmentation in degrees")
    parser.add_argument("--translate", type=float, default=None, help="Translation augmentation ratio")
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation ratio")
    parser.add_argument("--shear", type=float, default=None, help="Shear augmentation in degrees")
    parser.add_argument("--perspective", type=float, default=None, help="Perspective augmentation ratio")
    parser.add_argument("--flipud", type=float, default=None, help="Vertical flip probability")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip probability")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=None, help="MixUp augmentation probability")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize from YAML architecture instead of pretrained .pt weights")
    args = parser.parse_args()

    # Load configurations
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Override with command line args
    if args.model:
        model_cfg.model_name = args.model
    if args.output_dir:
        model_cfg.output_dir = args.output_dir
    if args.epochs:
        model_cfg.epochs = args.epochs
    if args.batch_size:
        model_cfg.batch_size = args.batch_size
    if args.imgsz is not None:
        data_cfg.image_size = args.imgsz
    if args.workers is not None:
        train_cfg.workers = args.workers
    if args.hsv_h is not None:
        train_cfg.hsv_h = args.hsv_h
    if args.hsv_s is not None:
        train_cfg.hsv_s = args.hsv_s
    if args.hsv_v is not None:
        train_cfg.hsv_v = args.hsv_v
    if args.degrees is not None:
        train_cfg.degrees = args.degrees
    if args.translate is not None:
        train_cfg.translate = args.translate
    if args.scale is not None:
        train_cfg.scale = args.scale
    if args.shear is not None:
        train_cfg.shear = args.shear
    if args.perspective is not None:
        train_cfg.perspective = args.perspective
    if args.flipud is not None:
        train_cfg.flipud = args.flipud
    if args.fliplr is not None:
        train_cfg.fliplr = args.fliplr
    if args.mosaic is not None:
        train_cfg.mosaic = args.mosaic
    if args.mixup is not None:
        train_cfg.mixup = args.mixup
    if args.from_scratch:
        model_cfg.pretrained = False

    if args.data_root:
        data_cfg.data_root = args.data_root
        data_cfg.yolo_dataset_dir = str(Path(args.data_root) / "yolo")
        data_cfg.data_yaml_path = str(Path(data_cfg.yolo_dataset_dir) / "data.yaml")

    # Handle dataset
    if args.data_yaml:
        data_yaml = args.data_yaml
    else:
        data_yaml = build_yolo_dataset_from_coco(
            coco_root=data_cfg.data_root,
            output_root=data_cfg.yolo_dataset_dir,
            preserve_category_ids=True,
        )

    print(f"Using data.yaml: {data_yaml}")

    # Load model. Passing a .pt checkpoint fine-tunes from pretrained weights.
    if model_cfg.pretrained:
        model = YOLO(model_cfg.model_name)
    else:
        model = YOLO(model_cfg.scratch_model_cfg)

    project_dir, run_name = resolve_ultralytics_save_args(model_cfg.output_dir)
    # Training configuration
    train_args = {
        "data": str(data_yaml),
        "epochs": model_cfg.epochs,
        "batch": model_cfg.batch_size,
        "workers": train_cfg.workers,
        "imgsz": data_cfg.image_size,
        "patience": model_cfg.patience,
        "device": train_cfg.device,
        "lr0": train_cfg.lr0,
        "lrf": train_cfg.lrf,
        "momentum": train_cfg.momentum,
        "weight_decay": train_cfg.weight_decay,
        "hsv_h": train_cfg.hsv_h,
        "hsv_s": train_cfg.hsv_s,
        "hsv_v": train_cfg.hsv_v,
        "degrees": train_cfg.degrees,
        "translate": train_cfg.translate,
        "scale": train_cfg.scale,
        "shear": train_cfg.shear,
        "perspective": train_cfg.perspective,
        "flipud": train_cfg.flipud,
        "fliplr": train_cfg.fliplr,
        "mosaic": train_cfg.mosaic,
        "mixup": train_cfg.mixup,
        "save_period": train_cfg.save_period,
        "project": project_dir,
        "name": run_name,
        "val": train_cfg.val,
        "split": train_cfg.split,
    }

    # Train the model
    print("Starting training...")
    print(f"Model source: {model_cfg.model_name if model_cfg.pretrained else model_cfg.scratch_model_cfg}")
    results = model.train(**train_args)

    print(f"Training completed. Results saved to {results.save_dir}")

    # Save best model path
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
