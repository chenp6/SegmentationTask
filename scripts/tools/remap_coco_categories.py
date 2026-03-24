"""
Remap COCO categories by category name to a target label spec.

This tool rewrites ``annotations[].category_id`` and the ``categories`` list
according to a name-based mapping file. It supports:
- preserving arbitrary target ids for external integrations
- optionally producing YOLO-compatible contiguous ids (0..N-1)

Example:
    python -m scripts.tools.remap_coco_categories \
      --input-root data/other_dataset \
      --output-root data/other_dataset_remapped \
      --mapping-spec remap_spec.json \
      --drop-empty-images

Single JSON mode:
    python -m scripts.tools.remap_coco_categories \
      --input-ann data/other_dataset/_annotations.coco.json \
      --output-ann data/other_dataset/_annotations.remapped.coco.json \
      --mapping-spec remap_spec.json

Example mapping spec:
{
  "categories": [
    {"name": "glove", "id": 10},
    {"name": "syringe", "id": 20},
    {"name": "needle", "id": 30}
  ]
}

YOLO-compatible output:
    python -m scripts.tools.remap_coco_categories \
      --input-root data/other_dataset \
      --output-root data/other_dataset_yolo_ids \
      --mapping-spec remap_spec.json \
      --make-contiguous-for-yolo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def anns_by_image_id(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
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


def normalize_name(value: str) -> str:
    return value.strip().casefold()


def build_target_categories(mapping_spec: dict, make_contiguous_for_yolo: bool) -> tuple[dict[str, dict], list[dict]]:
    raw_categories = mapping_spec.get("categories", [])
    if not isinstance(raw_categories, list) or not raw_categories:
        raise ValueError("mapping spec must contain a non-empty 'categories' list")

    normalized_items: list[dict] = []
    seen_names: set[str] = set()
    seen_ids: set[int] = set()

    for item in raw_categories:
        if "name" not in item or "id" not in item:
            raise ValueError("each mapping item must contain 'name' and 'id'")
        raw_name = str(item["name"])
        norm_name = normalize_name(raw_name)
        raw_id = int(item["id"])
        if norm_name in seen_names:
            raise ValueError(f"duplicate category name in mapping spec: {raw_name}")
        if raw_id in seen_ids:
            raise ValueError(f"duplicate category id in mapping spec: {raw_id}")
        seen_names.add(norm_name)
        seen_ids.add(raw_id)
        normalized_items.append(
            {
                "name": raw_name,
                "norm_name": norm_name,
                "requested_id": raw_id,
                "supercategory": item.get("supercategory", raw_name),
            }
        )

    if make_contiguous_for_yolo:
        normalized_items.sort(key=lambda item: item["requested_id"])
        for contiguous_id, item in enumerate(normalized_items):
            item["final_id"] = contiguous_id
    else:
        for item in normalized_items:
            item["final_id"] = item["requested_id"]

    target_by_name = {
        item["norm_name"]: {
            "id": int(item["final_id"]),
            "name": item["name"],
            "supercategory": item["supercategory"],
            "requested_id": int(item["requested_id"]),
        }
        for item in normalized_items
    }

    ordered_categories = [
        {
            "id": int(item["final_id"]),
            "name": item["name"],
            "supercategory": item["supercategory"],
        }
        for item in normalized_items
    ]
    ordered_categories.sort(key=lambda cat: int(cat["id"]))

    return target_by_name, ordered_categories


def remap_split(
    split: str,
    input_root: Path,
    output_root: Path,
    target_by_name: dict[str, dict],
    ordered_categories: list[dict],
    drop_empty_images: bool,
    force_copy_images: bool,
    write_annotations_only: bool,
    strict: bool,
) -> None:
    split_dir = input_root / split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"Skipping split '{split}': annotation file not found at {ann_path}")
        return

    coco = load_json(ann_path)
    original_images = coco.get("images", [])
    original_annotations = coco.get("annotations", [])
    original_categories = coco.get("categories", [])

    source_categories_by_id = {
        int(cat["id"]): cat for cat in original_categories
    }

    remapped_annotations: list[dict] = []
    skipped_annotations = 0
    missing_source_names: set[str] = set()

    for ann in original_annotations:
        source_cat = source_categories_by_id.get(int(ann.get("category_id", -1)))
        if source_cat is None:
            skipped_annotations += 1
            continue

        source_name = str(source_cat.get("name", "")).strip()
        target = target_by_name.get(normalize_name(source_name))
        if target is None:
            skipped_annotations += 1
            missing_source_names.add(source_name or f"id={ann.get('category_id')}")
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = int(target["id"])
        remapped_annotations.append(new_ann)

    if strict and missing_source_names:
        missing_list = ", ".join(sorted(missing_source_names))
        raise ValueError(f"Found source categories not present in mapping spec: {missing_list}")

    grouped_annotations = anns_by_image_id(remapped_annotations)
    if drop_empty_images:
        kept_images = [
            img for img in original_images
            if int(img["id"]) in grouped_annotations
        ]
    else:
        kept_images = list(original_images)

    kept_image_ids = {int(img["id"]) for img in kept_images}
    remapped_annotations = [
        ann for ann in remapped_annotations
        if int(ann["image_id"]) in kept_image_ids
    ]

    used_category_ids = {int(ann["category_id"]) for ann in remapped_annotations}
    kept_categories = [
        cat for cat in ordered_categories
        if int(cat["id"]) in used_category_ids
    ]

    out_split_dir = output_root / split
    copied_images = 0
    if not write_annotations_only:
        for image_info in kept_images:
            src_image_path = resolve_image_path(split_dir, image_info["file_name"])
            dst_image_path = out_split_dir / Path(image_info["file_name"]).name
            link_or_copy_image(src_image_path, dst_image_path, force_copy=force_copy_images)
            copied_images += 1

    new_coco = dict(coco)
    new_coco["images"] = kept_images
    new_coco["annotations"] = remapped_annotations
    new_coco["categories"] = kept_categories

    out_ann_path = out_split_dir / "_annotations.coco.json"
    save_json(out_ann_path, new_coco)

    print(f"\n[{split}] done")
    print(f"  images kept:         {len(kept_images)} / {len(original_images)}")
    print(f"  annotations kept:    {len(remapped_annotations)} / {len(original_annotations)}")
    print(f"  annotations skipped: {skipped_annotations}")
    print(f"  categories kept:     {len(kept_categories)} / {len(ordered_categories)}")
    print(f"  image files copied:  {copied_images}")
    if write_annotations_only:
        print("  image export:        skipped (--annotations-only)")
    if missing_source_names:
        print(f"  unmapped source names: {sorted(missing_source_names)}")
    print(f"  output:              {out_ann_path}")


def remap_single_annotation_file(
    input_ann: Path,
    output_ann: Path,
    target_by_name: dict[str, dict],
    ordered_categories: list[dict],
    drop_empty_images: bool,
    strict: bool,
) -> None:
    coco = load_json(input_ann)
    original_images = coco.get("images", [])
    original_annotations = coco.get("annotations", [])
    original_categories = coco.get("categories", [])

    source_categories_by_id = {
        int(cat["id"]): cat for cat in original_categories
    }

    remapped_annotations: list[dict] = []
    skipped_annotations = 0
    missing_source_names: set[str] = set()

    for ann in original_annotations:
        source_cat = source_categories_by_id.get(int(ann.get("category_id", -1)))
        if source_cat is None:
            skipped_annotations += 1
            continue

        source_name = str(source_cat.get("name", "")).strip()
        target = target_by_name.get(normalize_name(source_name))
        if target is None:
            skipped_annotations += 1
            missing_source_names.add(source_name or f"id={ann.get('category_id')}")
            continue

        new_ann = dict(ann)
        new_ann["category_id"] = int(target["id"])
        remapped_annotations.append(new_ann)

    if strict and missing_source_names:
        missing_list = ", ".join(sorted(missing_source_names))
        raise ValueError(f"Found source categories not present in mapping spec: {missing_list}")

    grouped_annotations = anns_by_image_id(remapped_annotations)
    if drop_empty_images:
        kept_images = [
            img for img in original_images
            if int(img["id"]) in grouped_annotations
        ]
    else:
        kept_images = list(original_images)

    kept_image_ids = {int(img["id"]) for img in kept_images}
    remapped_annotations = [
        ann for ann in remapped_annotations
        if int(ann["image_id"]) in kept_image_ids
    ]

    used_category_ids = {int(ann["category_id"]) for ann in remapped_annotations}
    kept_categories = [
        cat for cat in ordered_categories
        if int(cat["id"]) in used_category_ids
    ]

    new_coco = dict(coco)
    new_coco["images"] = kept_images
    new_coco["annotations"] = remapped_annotations
    new_coco["categories"] = kept_categories
    save_json(output_ann, new_coco)

    print("\n[single-json] done")
    print(f"  images kept:         {len(kept_images)} / {len(original_images)}")
    print(f"  annotations kept:    {len(remapped_annotations)} / {len(original_annotations)}")
    print(f"  annotations skipped: {skipped_annotations}")
    print(f"  categories kept:     {len(kept_categories)} / {len(ordered_categories)}")
    if missing_source_names:
        print(f"  unmapped source names: {sorted(missing_source_names)}")
    print(f"  output:              {output_ann}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap COCO categories by name to target ids")
    parser.add_argument("--input-root", default=None, help="Input COCO dataset root")
    parser.add_argument("--output-root", default=None, help="Output dataset root after remapping")
    parser.add_argument("--input-ann", default=None, help="Single input COCO annotation json path")
    parser.add_argument("--output-ann", default=None, help="Single output COCO annotation json path")
    parser.add_argument("--mapping-spec", required=True, help="Path to name-to-id mapping spec json")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits to process")
    parser.add_argument("--make-contiguous-for-yolo", action="store_true", help="Rewrite target ids into contiguous 0..N-1 ids for YOLO compatibility")
    parser.add_argument("--drop-empty-images", action="store_true", help="Drop images that have no annotations left after remapping")
    parser.add_argument("--copy-images", action="store_true", help="Copy files instead of hard-linking them into the output dataset")
    parser.add_argument("--annotations-only", action="store_true", help="Only write remapped _annotations.coco.json without linking/copying images")
    parser.add_argument("--strict", action="store_true", help="Fail if any source category name is missing from the mapping spec")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mapping_spec = load_json(Path(args.mapping_spec))
    target_by_name, ordered_categories = build_target_categories(
        mapping_spec=mapping_spec,
        make_contiguous_for_yolo=args.make_contiguous_for_yolo,
    )

    final_ids = [int(cat["id"]) for cat in ordered_categories]
    if not args.make_contiguous_for_yolo:
        sorted_ids = sorted(final_ids)
        contiguous_ids = list(range(len(sorted_ids)))
        if sorted_ids != contiguous_ids:
            print(
                "Warning: target ids are not contiguous 0..N-1. "
                "This is allowed for COCO, but raw YOLO class ids should be contiguous. "
                "Use --make-contiguous-for-yolo if this dataset will be used to train or evaluate YOLO directly."
            )

    single_json_mode = bool(args.input_ann or args.output_ann)
    if single_json_mode:
        if not args.input_ann or not args.output_ann:
            raise ValueError("--input-ann and --output-ann must be provided together")
        if args.input_root or args.output_root:
            raise ValueError("Do not combine --input-ann/--output-ann with --input-root/--output-root")
        remap_single_annotation_file(
            input_ann=Path(args.input_ann),
            output_ann=Path(args.output_ann),
            target_by_name=target_by_name,
            ordered_categories=ordered_categories,
            drop_empty_images=args.drop_empty_images,
            strict=args.strict,
        )
        return

    if not args.input_root or not args.output_root:
        raise ValueError("Either provide --input-root/--output-root or --input-ann/--output-ann")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        remap_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            target_by_name=target_by_name,
            ordered_categories=ordered_categories,
            drop_empty_images=args.drop_empty_images,
            force_copy_images=args.copy_images,
            write_annotations_only=args.annotations_only,
            strict=args.strict,
        )


if __name__ == "__main__":
    main()
