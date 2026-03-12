"""
Train YOLOv11-seg (Ultralytics) on parts segmentation. Uses YOLOv11 checkpoint (yolo11n-seg.pt etc.), not YOLO26.

Uses the dataset exported to YOLO-seg format (run export_yolo_seg_format.py first).
Starts from pretrained yolo11n-seg.pt (or yolo11s-seg.pt, etc.).

Requires: pip install ultralytics

Usage (from repo root):
  # 1. Export dataset once
  python -m scripts.parts_seg.export_yolo_seg_format --data_root data/ADE20KPart234 --output_dir data/ADE20KPart234_yolo_seg

  # 2. Train
  python -m scripts.parts_seg.train_yolo11_seg --data_yaml data/ADE20KPart234_yolo_seg/data.yaml --epochs 50
  python -m scripts.parts_seg.train_yolo11_seg --data_yaml data/ADE20KPart234_yolo_seg/data.yaml --model yolo11s-seg.pt --imgsz 512 --epochs 30
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11-seg on parts (Ultralytics)")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to data.yaml (from export_yolo_seg_format)")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Pretrained model: yolo11n-seg.pt, yolo11s-seg.pt, etc.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 avoids pin_memory connection errors)")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--project", type=str, default="output/parts_yolo11_seg")
    parser.add_argument("--name", type=str, default="train")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install Ultralytics: pip install ultralytics")

    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}. Run export_yolo_seg_format.py first.")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device or None,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print(f"Training done. Results: {results.save_dir}")


if __name__ == "__main__":
    main()
