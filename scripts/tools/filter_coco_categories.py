"""
Filter specific categories out of a COCO dataset.

This tool removes annotations whose ``category_id`` is in a given deny-list,
updates the ``categories`` section accordingly, and can optionally drop images
that no longer have any annotations after filtering.

Example:
    python -m scripts.tools.filter_coco_categories \
      --input-root data/medbin_dataset \
      --output-root data/medbin_dataset_filtered \
      --remove-category-ids 998 999 \
      --drop-empty-images
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_coco(ann_path: Path) -> dict:
    # Read one COCO annotation json file.
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(ann_path: Path, data: dict) -> None:
    # Write the filtered COCO annotation json.
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def anns_by_image_id(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    # Group annotations by image id for fast lookup.
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    # Resolve the source image path from the COCO file_name field.
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


def link_or_copy_image(src: Path, dst: Path, force_copy: bool = False) -> None:
    # Prefer hard-links, but fall back to copying when needed.
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def filter_split(
    split: str,
    input_root: Path,
    output_root: Path,
    remove_category_ids: set[int],
    drop_empty_images: bool,
    force_copy_images: bool,
) -> None:
    # Filter one split and write the result as a new COCO dataset split.
    split_dir = input_root / split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"Skipping split '{split}': annotation file not found at {ann_path}")
        return

    coco = load_coco(ann_path)
    original_images = coco.get("images", [])
    original_annotations = coco.get("annotations", [])
    original_categories = coco.get("categories", [])

    kept_annotations = [
        ann for ann in original_annotations
        if int(ann.get("category_id", -1)) not in remove_category_ids
    ]
    removed_annotations = len(original_annotations) - len(kept_annotations)

    kept_categories = [
        cat for cat in original_categories
        if int(cat.get("id", -1)) not in remove_category_ids
    ]

    grouped_annotations = anns_by_image_id(kept_annotations)
    if drop_empty_images:
        kept_images = [
            img for img in original_images
            if int(img["id"]) in grouped_annotations
        ]
    else:
        kept_images = list(original_images)

    kept_image_ids = {int(img["id"]) for img in kept_images}
    kept_annotations = [
        ann for ann in kept_annotations
        if int(ann["image_id"]) in kept_image_ids
    ]

    out_split_dir = output_root / split
    copied_images = 0
    for image_info in kept_images:
        src_image_path = resolve_image_path(split_dir, image_info["file_name"])
        dst_image_path = out_split_dir / Path(image_info["file_name"]).name
        link_or_copy_image(src_image_path, dst_image_path, force_copy=force_copy_images)
        copied_images += 1

    new_coco = dict(coco)
    new_coco["images"] = kept_images
    new_coco["annotations"] = kept_annotations
    new_coco["categories"] = kept_categories

    out_ann_path = out_split_dir / "_annotations.coco.json"
    save_coco(out_ann_path, new_coco)

    print(f"\n[{split}] done")
    print(f"  removed category ids: {sorted(remove_category_ids)}")
    print(f"  images kept:          {len(kept_images)} / {len(original_images)}")
    print(f"  annotations kept:     {len(kept_annotations)} / {len(original_annotations)}")
    print(f"  annotations removed:  {removed_annotations}")
    print(f"  categories kept:      {len(kept_categories)} / {len(original_categories)}")
    print(f"  image files copied:   {copied_images}")
    print(f"  output:               {out_ann_path}")


def parse_args() -> argparse.Namespace:
    # CLI entry point for selecting splits and category ids to remove.
    parser = argparse.ArgumentParser(description="Filter categories out of a COCO dataset")
    parser.add_argument("--input-root", required=True, help="Input COCO dataset root")
    parser.add_argument("--output-root", required=True, help="Output dataset root after filtering")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits to process")
    parser.add_argument(
        "--remove-category-ids",
        type=int,
        nargs="+",
        required=True,
        help="One or more COCO category ids to remove, e.g. 998 999",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Drop images that have no annotations left after filtering",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy files instead of hard-linking them into the output dataset",
    )
    return parser.parse_args()


def main() -> None:
    # Run the filtering split by split.
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    remove_category_ids = set(args.remove_category_ids)
    for split in args.splits:
        filter_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            remove_category_ids=remove_category_ids,
            drop_empty_images=args.drop_empty_images,
            force_copy_images=args.copy_images,
        )


if __name__ == "__main__":
    main()
