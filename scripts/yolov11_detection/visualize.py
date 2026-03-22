"""
Visualize YOLOv11 object detection predictions.

Usage:
    python -m scripts.yolov11_detection.visualize --model output/yolov11_detection/exp/weights/best.pt --source data/hiod_coco/test --max-images 10
"""
import argparse
import ast
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]


def _collect_image_paths(source_path: Path, max_images: int) -> list[Path]:
    if source_path.is_dir():
        image_paths: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(source_path.glob(f"**/*{ext}"))
        image_paths = sorted({path.resolve() for path in image_paths})
        if max_images > 0:
            image_paths = random.sample(image_paths, min(max_images, len(image_paths)))
        return [Path(p) for p in image_paths]
    return [source_path]


def _make_color(seed: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    color = rng.integers(64, 255, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def _load_simple_data_yaml(yaml_path: Path) -> dict:
    data: dict = {}
    names: list[str] = []

    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- "):
            names.append(line[2:].strip())
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = None
            continue

        if value.startswith("[") and value.endswith("]"):
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value
        else:
            data[key] = value

    if names:
        data["names"] = names

    return data


def _resolve_split_images_dir(data_cfg: dict, split: str, yaml_path: Path) -> Path:
    path_root = Path(str(data_cfg.get("path", yaml_path.parent)))
    split_value = data_cfg.get(split)
    if not split_value:
        raise KeyError(f"Split '{split}' not found in {yaml_path}")
    return (path_root / str(split_value)).resolve()


def _label_path_from_image(image_path: Path) -> Path:
    split_dir = image_path.parent.parent
    labels_dir = split_dir / "labels"
    return labels_dir / f"{image_path.stem}.txt"


def _draw_ground_truth(
    image_bgr: np.ndarray,
    label_path: Path,
    class_names: list[str],
) -> np.ndarray:
    canvas = image_bgr.copy()
    image_h, image_w = canvas.shape[:2]

    if not label_path.exists():
        return canvas

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        x_center = float(parts[1]) * image_w
        y_center = float(parts[2]) * image_h
        box_w = float(parts[3]) * image_w
        box_h = float(parts[4]) * image_h

        x1 = int(x_center - box_w / 2.0)
        y1 = int(y_center - box_h / 2.0)
        x2 = int(x_center + box_w / 2.0)
        y2 = int(y_center + box_h / 2.0)

        color = _make_color(class_id * 97 + 11)
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{class_name} gt",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return canvas


def _visualize_ground_truth(
    data_yaml: Path,
    split: str,
    max_images: int,
    save: bool,
    show: bool,
    output_dir: Path,
) -> None:
    data_cfg = _load_simple_data_yaml(data_yaml)
    class_names = data_cfg.get("names", [])
    images_dir = _resolve_split_images_dir(data_cfg, split=split, yaml_path=data_yaml)
    image_paths = _collect_image_paths(images_dir, max_images)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {len(image_paths)} ground-truth images from {images_dir}")

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        label_path = _label_path_from_image(image_path)
        rendered = _draw_ground_truth(image_bgr, label_path=label_path, class_names=class_names)

        if save:
            out_path = output_dir / image_path.name
            cv2.imwrite(str(out_path), rendered)

        if show:
            cv2.imshow("YOLOv11 Detection Ground Truth", rendered)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    if save:
        print(f"Ground-truth visualizations saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLOv11 object detection predictions")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model weights")
    parser.add_argument("--source", type=str, default=None, help="Path to images, folder, or video")
    parser.add_argument("--max-images", type=int, default=10, help="Maximum number of images to process")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--save", action="store_true", help="Save prediction results")
    parser.add_argument("--show", action="store_true", help="Show results in window")
    parser.add_argument("--ground-truth", action="store_true", help="Visualize YOLO ground-truth labels instead of predictions")
    parser.add_argument("--data-yaml", type=str, default=None, help="data.yaml path used for ground-truth visualization")
    parser.add_argument("--split", type=str, default="val", help="Dataset split for ground-truth mode: train/val/test")
    parser.add_argument("--output-dir", type=str, default="output/yolov11_detection_gt", help="Output directory for ground-truth mode")
    args = parser.parse_args()

    if args.ground_truth:
        if not args.data_yaml:
            raise ValueError("--data-yaml is required when using --ground-truth")
        _visualize_ground_truth(
            data_yaml=Path(args.data_yaml),
            split=args.split,
            max_images=args.max_images,
            save=args.save,
            show=args.show,
            output_dir=Path(args.output_dir),
        )
        return

    if not args.model or not args.source:
        raise ValueError("--model and --source are required unless --ground-truth is used")

    model = YOLO(args.model)
    source_path = Path(args.source)

    if source_path.is_dir():
        image_paths = _collect_image_paths(source_path, args.max_images)
        source_arg = [str(path) for path in image_paths]
    else:
        image_paths = [source_path]
        source_arg = str(source_path)

    print(f"Processing {len(image_paths)} inputs...")

    results = model.predict(
        source=source_arg,
        conf=args.conf,
        device=args.device,
        save=args.save,
        show=args.show,
        max_det=100,
    )

    total_detections = sum(len(result.boxes) for result in results)
    print(f"Total detections: {total_detections}")

    if args.save:
        print("Prediction images/videos were saved to the Ultralytics output directory shown above.")


if __name__ == "__main__":
    main()
