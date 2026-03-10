"""
Dataset utilities for YOLOv11 segmentation training.

Builds a YOLO-seg compatible view from the same COCO dataset used by
Mask2Former/SAM pipelines under ``data/hospital_coco``.
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


def _category_metadata(categories: List[dict]) -> Tuple[Dict[int, int], List[str]]:
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}
    names = [cat["name"] for cat in sorted_cats]
    return cat_id_to_idx, names


def _normalize_polygon(segmentation: List[float], width: int, height: int) -> str:
    coords = []
    for i in range(0, len(segmentation), 2):
        x = min(max(segmentation[i] / width, 0.0), 1.0)
        y = min(max(segmentation[i + 1] / height, 0.0), 1.0)
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return " ".join(coords)


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

        segmentation = ann.get("segmentation", [])
        if not isinstance(segmentation, list):
            continue

        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
                continue
            polygon_str = _normalize_polygon(polygon, width, height)
            lines.append(f"{cat_id_to_idx[ann['category_id']]} {polygon_str}")

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


def build_yolo_dataset_from_coco(
    coco_root: str | Path,
    output_root: str | Path | None = None,
    output_yaml: str = "data.yaml",
) -> Path:
    """
    Convert the COCO dataset used by mask2former into a YOLO-seg dataset.

    Expected COCO layout:
      data/hospital_coco/
        train/_annotations.coco.json
        valid/_annotations.coco.json
        test/_annotations.coco.json
    """
    coco_root = Path(coco_root)
    if output_root is None:
        output_root = coco_root / "yolo"
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_mapping = {"train": "train", "valid": "val", "test": "test"}
    class_names = None

    for coco_split, yolo_split in split_mapping.items():
        ann_path = coco_root / coco_split / "_annotations.coco.json"
        if not ann_path.exists():
            if coco_split == "test":
                continue
            raise FileNotFoundError(f"Annotation not found: {ann_path}")

        coco = load_coco(ann_path)
        cat_id_to_idx, names = _category_metadata(coco["categories"])
        if class_names is None:
            class_names = names

        images_by_id = {img["id"]: img for img in coco["images"]}
        anns_by_image: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        split_images_dir = output_root / yolo_split / "images"
        split_labels_dir = output_root / yolo_split / "labels"

        for image_id, image_info in images_by_id.items():
            file_name = image_info["file_name"]
            src_image = coco_root / coco_split / file_name
            if not src_image.exists() or src_image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            dst_image = split_images_dir / Path(file_name).name
            _link_or_copy_image(src_image, dst_image)

            label_path = split_labels_dir / f"{Path(file_name).stem}.txt"
            _write_label_file(
                label_path=label_path,
                anns=anns_by_image.get(image_id, []),
                cat_id_to_idx=cat_id_to_idx,
                width=image_info["width"],
                height=image_info["height"],
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

    print(f"Prepared YOLO dataset at: {output_root}")
    print(f"YOLO data.yaml: {yaml_path}")
    return yaml_path
