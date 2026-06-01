"""
Convert YOLO segmentation TXT predictions to Roboflow-style COCO dataset layout.

Input:
  - images directory
  - labels directory from YOLO predict/segment (one .txt per image)

Output:
  output_root/
    <split>/               # default: train
      *.jpg|png
      _annotations.coco.json

Example:
  python -m scripts.tools.yolo_seg_txt_to_coco \
    --images-dir data/photos_med_0507 \
    --labels-dir runs/segment/photos_med_0507_degress180/labels \
    --output-root data/photos_med_0507_pseudo \
    --split train \
    --class-names-file data/medbin_dataset/yolo_segmentation/data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from pycocotools import mask as coco_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _polygon_area(points_xy: list[float]) -> float:
    xs = points_xy[0::2]
    ys = points_xy[1::2]
    n = len(xs)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        area2 += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area2) * 0.5


def _polygon_bbox(points_xy: list[float]) -> list[float]:
    xs = points_xy[0::2]
    ys = points_xy[1::2]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def _parse_yolo_seg_line(line: str, width: int, height: int, conf_thresh: float) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None

    cls_id = int(float(parts[0]))
    vals = [float(x) for x in parts[1:]]

    # YOLO predict labels often append confidence as the last value.
    if len(vals) % 2 == 1:
        conf = vals[-1]
        if conf < conf_thresh:
            return None
        vals = vals[:-1]

    if len(vals) < 6 or len(vals) % 2 != 0:
        return None

    abs_xy: list[float] = []
    for i in range(0, len(vals), 2):
        x = min(max(vals[i], 0.0), 1.0) * width
        y = min(max(vals[i + 1], 0.0), 1.0) * height
        abs_xy.extend([float(x), float(y)])
    return cls_id, abs_xy


def _polygon_to_rle(points_xy: list[float], width: int, height: int) -> dict:
    rles = coco_mask.frPyObjects([points_xy], height, width)
    rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _load_class_names(
    class_names: str | None,
    class_names_file: str | None,
    max_cls_id: int,
) -> list[str]:
    if class_names:
        names = [x.strip() for x in class_names.split(",") if x.strip()]
        if names:
            if len(names) <= max_cls_id:
                names.extend([f"class_{i}" for i in range(len(names), max_cls_id + 1)])
            return names

    if class_names_file:
        p = Path(class_names_file)
        if p.suffix.lower() in {".json"}:
            payload = json.loads(p.read_text(encoding="utf-8"))
            names = payload.get("names", [])
            if isinstance(names, list) and names:
                if len(names) <= max_cls_id:
                    names.extend([f"class_{i}" for i in range(len(names), max_cls_id + 1)])
                return [str(n) for n in names]
        else:
            # light parser for YOLO data.yaml names section:
            # names:
            #   - cls0
            #   - cls1
            lines = p.read_text(encoding="utf-8").splitlines()
            names: list[str] = []
            in_names = False
            for ln in lines:
                s = ln.strip()
                if s.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if s.startswith("- "):
                        names.append(s[2:].strip())
                    elif s and not s.startswith("#"):
                        break
            if names:
                if len(names) <= max_cls_id:
                    names.extend([f"class_{i}" for i in range(len(names), max_cls_id + 1)])
                return names

    return [f"class_{i}" for i in range(max_cls_id + 1)]


def convert(
    images_dir: str,
    labels_dir: str,
    output_root: str,
    split: str,
    conf_thresh: float,
    class_names: str | None,
    class_names_file: str | None,
    segmentation_format: str,
) -> None:
    images_path = Path(images_dir).resolve()
    labels_path = Path(labels_dir).resolve()
    split_dir = Path(output_root).resolve() / split
    split_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_path}")

    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1
    max_cls_id = 0

    for img_id, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as img:
            w, h = img.size

        dst_img_name = img_path.name
        shutil.copy2(img_path, split_dir / dst_img_name)

        images.append(
            {
                "id": img_id,
                "file_name": dst_img_name,
                "width": int(w),
                "height": int(h),
            }
        )

        label_file = labels_path / f"{img_path.stem}.txt"
        if not label_file.exists():
            continue

        for line in label_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_yolo_seg_line(line, w, h, conf_thresh=conf_thresh)
            if parsed is None:
                continue

            cls_id, polygon_xy = parsed
            max_cls_id = max(max_cls_id, cls_id)
            bbox = _polygon_bbox(polygon_xy)
            area = _polygon_area(polygon_xy)
            if area <= 0:
                continue

            if segmentation_format == "rle":
                segmentation = _polygon_to_rle(polygon_xy, w, h)
            else:
                segmentation = [polygon_xy]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls_id),
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    names = _load_class_names(class_names, class_names_file, max_cls_id=max_cls_id)
    categories = [{"id": i, "name": names[i], "supercategory": names[i]} for i in range(len(names))]

    coco = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    out_ann = split_dir / "_annotations.coco.json"
    out_ann.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] images={len(images)}, annotations={len(annotations)}, categories={len(categories)}")
    print(f"[done] output split: {split_dir}")
    print(f"[done] annotation: {out_ann}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation txt predictions to COCO JSON.")
    parser.add_argument("--images-dir", required=True, help="Source images directory")
    parser.add_argument("--labels-dir", required=True, help="YOLO segmentation labels directory")
    parser.add_argument("--output-root", required=True, help="Output dataset root")
    parser.add_argument("--split", default="train", help="Output split name, e.g. train or valid")
    parser.add_argument("--conf-thresh", type=float, default=0.0, help="Drop polygons with confidence < threshold")
    parser.add_argument("--class-names", default=None, help="Comma-separated class names by id order")
    parser.add_argument("--class-names-file", default=None, help="Path to data.yaml/json containing names")
    parser.add_argument(
        "--segmentation-format",
        choices=["rle", "polygon"],
        default="rle",
        help="Output COCO segmentation format (default: rle)",
    )
    args = parser.parse_args()

    convert(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_root=args.output_root,
        split=args.split,
        conf_thresh=args.conf_thresh,
        class_names=args.class_names,
        class_names_file=args.class_names_file,
        segmentation_format=args.segmentation_format,
    )


if __name__ == "__main__":
    main()
