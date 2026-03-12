"""
Evaluate YOLOv11-seg hierarchy model (part segmentation) on validation set.

Loads a trained checkpoint from train_yolo_seg_hierarchy and runs Ultralytics
segment validation (box/mask mAP, etc.). Uses the same data.yaml as training.

To match training validation metrics, use the same imgsz/batch as training (or omit
--imgsz/--batch to auto-use values saved in the checkpoint).

Usage (from repo root):
  python -m scripts.parts_seg.evaluate_yolo_seg_hierarchy --checkpoint .../weights/best.pt --data_yaml data/ADE20KPart234_yolo_seg/data.yaml
  python -m scripts.parts_seg.evaluate_yolo_seg_hierarchy --checkpoint .../weights/best.pt --data_yaml data/ADE20KPart234_yolo_seg/data.yaml --imgsz 512 --batch 4
"""
import argparse
from pathlib import Path

from .model_yolo_seg_hierarchy import load_yolo_seg_hierarchy_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchy YOLOv11-seg (part segmentation)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt from hierarchy training (e.g. .../weights/last.pt)")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to data.yaml (same as training)")
    parser.add_argument("--split", type=str, default="val", help="Data split to evaluate (default: val)")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (default: from checkpoint train_args, else 640)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (default: from checkpoint train_args, else 4)")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    print(f"Loading hierarchy model from {args.checkpoint}")
    model, ckpt = load_yolo_seg_hierarchy_from_checkpoint(
        args.checkpoint,
        data_yaml=str(data_yaml),
        device=args.device or None,
    )

    # Match training validation: use same imgsz/batch as in train_args when not overridden
    train_args = ckpt.get("train_args") or {}
    imgsz = args.imgsz if args.imgsz is not None else train_args.get("imgsz", 640)
    batch = args.batch if args.batch is not None else train_args.get("batch", 4)

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        batch=batch,
        imgsz=imgsz,
        device=args.device or None,
    )
    if metrics is not None:
        print("\nValidation complete.")
        if hasattr(metrics, "box") and metrics.box is not None:
            print(f"  Box mAP50:    {getattr(metrics.box, 'map50', 0):.4f}")
            print(f"  Box mAP50-95: {getattr(metrics.box, 'map', 0):.4f}")
        if hasattr(metrics, "seg") and metrics.seg is not None:
            print(f"  Seg mAP50:    {getattr(metrics.seg, 'map50', 0):.4f}")
            print(f"  Seg mAP50-95: {getattr(metrics.seg, 'map', 0):.4f}")


if __name__ == "__main__":
    main()
