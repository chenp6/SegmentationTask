"""
Convert COCO segmentation annotations from polygon format to RLE format.

Supports Roboflow-style dataset layout:
  input_root/
    train/_annotations.coco.json
    valid/_annotations.coco.json
    test/_annotations.coco.json

If --output-root is omitted, files are updated in-place.

Example:
  python -m scripts.tools.coco_polygon_to_rle \
    --input-root data/medbin_dataset \
    --output-root data/medbin_dataset_rle
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _normalize_rle_dict(rle: dict) -> dict:
    out = dict(rle)
    if isinstance(out.get("counts"), bytes):
        out["counts"] = out["counts"].decode("utf-8")
    return out


def _polygon_to_rle(segmentation: list, height: int, width: int) -> dict | None:
    if not segmentation:
        return None
    rles = coco_mask.frPyObjects(segmentation, height, width)
    if isinstance(rles, list):
        rle = coco_mask.merge(rles)
    else:
        rle = rles
    return _normalize_rle_dict(rle)


def _ensure_file_name_basename(coco: dict) -> None:
    for img in coco.get("images", []):
        file_name = img.get("file_name")
        if isinstance(file_name, str):
            img["file_name"] = Path(file_name).name


def _copy_images_if_needed(src_split: Path, dst_split: Path) -> int:
    copied = 0
    dst_split.mkdir(parents=True, exist_ok=True)
    for p in src_split.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(p, dst_split / p.name)
            copied += 1
    return copied


def convert_split(src_ann: Path, dst_ann: Path, min_area: float = 1.0) -> tuple[int, int, int]:
    coco = _load_json(src_ann)
    _ensure_file_name_basename(coco)
    images_by_id = {int(img["id"]): img for img in coco.get("images", [])}

    converted = 0
    skipped = 0
    already_rle = 0

    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if seg is None:
            skipped += 1
            continue

        if isinstance(seg, dict):
            ann["segmentation"] = _normalize_rle_dict(seg)
            already_rle += 1
            continue

        if not isinstance(seg, list):
            skipped += 1
            continue

        image_id = int(ann["image_id"])
        img = images_by_id.get(image_id)
        if img is None:
            skipped += 1
            continue

        h = int(img["height"])
        w = int(img["width"])
        rle = _polygon_to_rle(seg, h, w)
        if rle is None:
            skipped += 1
            continue

        # Keep area/bbox if present. Optionally refresh area from mask when missing/invalid.
        if float(ann.get("area", 0.0)) <= 0:
            decoded = coco_mask.decode(rle)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            ann["area"] = float(decoded.astype(np.uint8).sum())
            if ann["area"] < min_area:
                skipped += 1
                continue

        ann["segmentation"] = rle
        converted += 1

    _save_json(dst_ann, coco)
    return converted, already_rle, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO polygon segmentation to RLE.")
    parser.add_argument("--input-root", required=True, help="COCO dataset root")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output dataset root. Omit to update annotations in-place.",
    )
    parser.add_argument(
        "--splits",
        default="train,valid,test",
        help="Comma-separated split names to process",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="When --output-root is set, also copy split images into output root",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else None

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No split specified. Use --splits train,valid,test")

    for split in splits:
        src_split = input_root / split
        src_ann = src_split / "_annotations.coco.json"
        if not src_ann.exists():
            print(f"[skip] {split}: annotation not found -> {src_ann}")
            continue

        if output_root is None:
            dst_split = src_split
        else:
            dst_split = output_root / split
            dst_split.mkdir(parents=True, exist_ok=True)
            if args.copy_images:
                copied = _copy_images_if_needed(src_split, dst_split)
                print(f"[{split}] copied images: {copied}")

        dst_ann = dst_split / "_annotations.coco.json"
        converted, already_rle, skipped = convert_split(src_ann, dst_ann)
        print(
            f"[{split}] done -> {dst_ann} | converted={converted}, "
            f"already_rle={already_rle}, skipped={skipped}"
        )


if __name__ == "__main__":
    main()

