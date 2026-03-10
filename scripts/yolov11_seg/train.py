"""
Train YOLOv11 segmentation model using Ultralytics on the shared COCO dataset.

Usage:
    python -m scripts.yolov11_seg.train
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from .config import DataConfig, ModelConfig, TrainConfig
from .dataset import build_yolo_dataset_from_coco


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on the shared COCO dataset")
    parser.add_argument("--data-yaml", type=str, default=None,
                       help="Path to an existing YOLO data.yaml")
    parser.add_argument("--data-root", type=str, default=None,
                       help="COCO dataset root shared with mask2former, e.g. data/hospital_coco")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name/size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--from-scratch", action="store_true",
                       help="Initialize from YAML architecture instead of pretrained .pt weights")
    args = parser.parse_args()

    # Load configurations
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Override with command line args
    if args.model:
        model_cfg.model_name = args.model
    if args.epochs:
        model_cfg.epochs = args.epochs
    if args.batch_size:
        model_cfg.batch_size = args.batch_size
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
        )

    print(f"Using data.yaml: {data_yaml}")

    # Load model. Passing a .pt checkpoint fine-tunes from pretrained weights.
    if model_cfg.pretrained:
        model = YOLO(model_cfg.model_name)
    else:
        model = YOLO(model_cfg.scratch_model_cfg)

    # Training configuration
    train_args = {
        "data": str(data_yaml),
        "epochs": model_cfg.epochs,
        "batch": model_cfg.batch_size,
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
        "project": model_cfg.output_dir,
        "name": train_cfg.name,
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
