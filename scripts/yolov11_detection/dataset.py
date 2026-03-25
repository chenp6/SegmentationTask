"""
Dataset utilities for YOLOv11 object detection.

Builds a YOLO detection dataset view from the original Roboflow COCO detection
"""
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_coco(ann_path: Path) -> dict:
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _category_metadata(
    categories: List[dict],
    preserve_category_ids: bool = True,
) -> Tuple[Dict[int, int], List[str]]:
    sorted_cats = sorted(categories, key=lambda c: c["id"])

    if not preserve_category_ids:
        cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
        names = [cat["name"] for cat in sorted_cats]
        return cat_id_to_idx, names

    max_category_id = max(int(cat["id"]) for cat in sorted_cats)
    names = [f"unused_class_{idx}" for idx in range(max_category_id + 1)]
    cat_id_to_idx: Dict[int, int] = {}
    for cat in sorted_cats:
        cat_id = int(cat["id"])
        cat_id_to_idx[cat_id] = cat_id
        names[cat_id] = str(cat["name"])
    return cat_id_to_idx, names


def _resolve_image_path(split_dir: Path, file_name: str) -> Path:
    direct = split_dir / file_name
    if direct.exists():
        return direct

    by_name = split_dir / Path(file_name).name
    if by_name.exists():
        return by_name

    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found under {split_dir}: {file_name}")


def _normalize_bbox_xywh(bbox: List[float], width: int, height: int) -> str:
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(0.0, min(x, width))
    y1 = max(0.0, min(y, height))
    w = max(0.0, min(w, width - x1))
    h = max(0.0, min(h, height - y1))

    x_center = (x1 + w / 2.0) / width
    y_center = (y1 + h / 2.0) / height
    w_norm = w / width
    h_norm = h / height

    return f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def _write_label_file(
    label_path: Path,
    anns: List[dict],
    cat_id_to_idx: Dict[int, int],
    width: int,
    height: int,
) -> None:
    lines = []
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        if "bbox" not in ann:
            continue

        bbox_str = _normalize_bbox_xywh(ann["bbox"], width=width, height=height)
        lines.append(f"{cat_id_to_idx[ann['category_id']]} {bbox_str}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _link_or_copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_yolo_dataset_from_coco_detection(
    coco_root: str | Path,
    output_root: str | Path | None = None,
    output_yaml: str = "data.yaml",
    preserve_category_ids: bool = True,
) -> Path:
    """
    Convert the original COCO detection dataset into a YOLO detection dataset.

    Expected layout:
      data/hiod_coco/
        train/_annotations.coco.json
        valid/_annotations.coco.json
        test/_annotations.coco.json
    """
    coco_root = Path(coco_root)
    if output_root is None:
        output_root = coco_root / "yolo_detection"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_mapping = {"train": "train", "valid": "val", "test": "test"}
    class_names = None

    for coco_split, yolo_split in split_mapping.items():
        split_dir = coco_root / coco_split
        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.exists():
            if coco_split == "test":
                continue
            raise FileNotFoundError(f"Annotation not found: {ann_path}")

        coco = load_coco(ann_path)
        cat_id_to_idx, names = _category_metadata(
            coco["categories"],
            preserve_category_ids=preserve_category_ids,
        )
        if class_names is None:
            class_names = names
        elif class_names != names:
            raise RuntimeError(
                f"Category definition mismatch between splits under {coco_root}. "
                "Expected the same categories ordering/names across all processed splits."
            )

        images_by_id = {img["id"]: img for img in coco["images"]}
        anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        split_images_dir = output_root / yolo_split / "images"
        split_labels_dir = output_root / yolo_split / "labels"

        for image_id, image_info in images_by_id.items():
            src_image = _resolve_image_path(split_dir, image_info["file_name"])
            dst_image = split_images_dir / Path(src_image.name)
            _link_or_copy_image(src_image, dst_image)

            label_path = split_labels_dir / f"{dst_image.stem}.txt"
            _write_label_file(
                label_path=label_path,
                anns=anns_by_image.get(image_id, []),
                cat_id_to_idx=cat_id_to_idx,
                width=int(image_info["width"]),
                height=int(image_info["height"]),
            )

    if class_names is None:
        raise RuntimeError(f"No COCO categories found under {coco_root}")

    yaml_path = output_root / output_yaml
    yaml_lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(class_names)}",
        "names:",
    ]
    yaml_lines.extend([f"  - {name}" for name in class_names])

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"Prepared YOLO detection dataset at: {output_root}")
    print(f"YOLO data.yaml: {yaml_path}")
    return yaml_path
