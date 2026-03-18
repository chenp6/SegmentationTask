"""
Evaluate a trained YOLOv11 object detection model.

Usage:
    python -m scripts.yolov11_detection.evaluate --model output/yolov11_detection/exp/weights/best.pt --data data.yaml
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 object detection model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    args = parser.parse_args()

    model = YOLO(args.model)

    print(f"Evaluating model on {args.split} set...")
    results = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch_size,
        device=args.device,
        save_json=True,
        save_txt=True,
        save_conf=True,
        save=True,
    )

    print("\nEvaluation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
