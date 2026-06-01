"""
Remove selected classes from a COCO annotation file.

This script removes:
1. Categories whose names are listed in --remove-class
2. Annotations belonging to removed categories

It keeps image entries unchanged (even if an image ends up with zero annotations).

Usage:
  python -m scripts.tools.remove_coco_classes \
    --input-json data/myset/train/_annotations.coco.json \
    --output-json data/myset/train/_annotations.filtered.coco.json \
    --remove-class class_a \
    --remove-class class_b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove selected classes from COCO annotations")
    parser.add_argument("--input-json", required=True, help="Input COCO annotation json")
    parser.add_argument("--output-json", required=True, help="Output COCO annotation json")
    parser.add_argument(
        "--remove-class",
        action="append",
        default=[],
        help="Class name to remove (repeatable)",
    )
    parser.add_argument(
        "--drop-empty-images",
        action="store_true",
        help="Also remove images that have no annotations after filtering",
    )
    parser.add_argument(
        "--resort-category-ids",
        action="store_true",
        help="Resort/reindex kept category ids to a contiguous range starting from 1",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resort_category_ids(coco: dict) -> dict:
    categories = coco.get("categories", [])
    annotations = coco.get("annotations", [])

    sorted_categories = sorted(categories, key=lambda c: int(c.get("id", 0)))
    old_to_new_id = {int(cat["id"]): new_id for new_id, cat in enumerate(sorted_categories, start=1)}

    remapped_categories = []
    for cat in sorted_categories:
        new_cat = dict(cat)
        new_cat["id"] = old_to_new_id[int(cat["id"])]
        remapped_categories.append(new_cat)

    remapped_annotations = []
    for ann in annotations:
        old_cat_id = int(ann.get("category_id", -1))
        if old_cat_id not in old_to_new_id:
            continue
        new_ann = dict(ann)
        new_ann["category_id"] = old_to_new_id[old_cat_id]
        remapped_annotations.append(new_ann)

    out = dict(coco)
    out["categories"] = remapped_categories
    out["annotations"] = remapped_annotations
    return out


def filter_coco(
    coco: dict, remove_names: set[str], drop_empty_images: bool, resort_category_ids_enabled: bool
) -> dict:
    categories = coco.get("categories", [])
    annotations = coco.get("annotations", [])
    images = coco.get("images", [])

    kept_categories = [cat for cat in categories if str(cat.get("name", "")) not in remove_names]
    kept_category_ids = {int(cat["id"]) for cat in kept_categories}

    kept_annotations = [
        ann for ann in annotations if int(ann.get("category_id", -1)) in kept_category_ids
    ]

    if drop_empty_images:
        kept_image_ids = {int(ann["image_id"]) for ann in kept_annotations}
        kept_images = [img for img in images if int(img.get("id", -1)) in kept_image_ids]
    else:
        kept_images = images

    out = dict(coco)
    out["categories"] = kept_categories
    out["annotations"] = kept_annotations
    out["images"] = kept_images
    if resort_category_ids_enabled:
        out = resort_category_ids(out)
    return out


def main() -> None:
    args = parse_args()
    if not args.remove_class:
        raise ValueError("Please provide at least one --remove-class")

    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    coco = load_json(input_path)
    remove_names = {name.strip() for name in args.remove_class if name.strip()}
    filtered = filter_coco(
        coco,
        remove_names,
        args.drop_empty_images,
        args.resort_category_ids,
    )
    save_json(output_path, filtered)

    print("Class removal completed.")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print("Removed classes:")
    for name in sorted(remove_names):
        print(f"  - {name}")
    print(f"Categories:  {len(coco.get('categories', []))} -> {len(filtered.get('categories', []))}")
    print(f"Annotations: {len(coco.get('annotations', []))} -> {len(filtered.get('annotations', []))}")
    print(f"Images:      {len(coco.get('images', []))} -> {len(filtered.get('images', []))}")
    print(f"Resort category ids: {'enabled' if args.resort_category_ids else 'disabled'}")


if __name__ == "__main__":
    main()
