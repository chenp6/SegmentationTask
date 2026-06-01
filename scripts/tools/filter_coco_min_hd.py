"""
Filter a COCO json by minimum image resolution (HD by default).

This tool keeps only images whose resolution is >= min_width x min_height,
removes annotations linked to filtered-out images, writes a new COCO json,
and copies kept image files to a new folder.

Example:
  python -m scripts.tools.filter_coco_min_hd \
    --input-json data/myset/train/_annotations.coco.json \
    --image-root data/myset/train \
    --output-json data/myset_hd/train/_annotations.coco.json \
    --output-image-dir data/myset_hd/train/images
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    direct = image_root / file_name
    if direct.exists():
        return direct

    by_name = image_root / Path(file_name).name
    if by_name.exists():
        return by_name

    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = image_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found under {image_root}: {file_name}")


def copy_kept_images(kept_images: list[dict], image_root: Path, output_image_dir: Path) -> tuple[int, int]:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0

    for img in kept_images:
        file_name = str(img.get("file_name", ""))
        if not file_name:
            missing += 1
            continue

        rel = Path(file_name)
        if rel.is_absolute() or ".." in rel.parts:
            # Skip unsafe paths to avoid writing outside output_image_dir.
            missing += 1
            continue

        try:
            src = resolve_image_path(image_root, file_name)
        except FileNotFoundError:
            missing += 1
            continue

        dst = output_image_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            # Keep first file when duplicate path appears.
            continue

        shutil.copy2(src, dst)
        copied += 1

    return copied, missing


def filter_coco_by_resolution(
    coco: dict,
    min_width: int,
    min_height: int,
    allow_rotated: bool,
) -> tuple[list[dict], list[dict], set[int], int]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    kept_images: list[dict] = []
    kept_image_ids: set[int] = set()
    missing_wh = 0

    for img in images:
        w = img.get("width")
        h = img.get("height")
        if w is None or h is None:
            missing_wh += 1
            continue

        w_i = int(w)
        h_i = int(h)

        ok = (w_i >= min_width and h_i >= min_height)
        if allow_rotated:
            ok = ok or (h_i >= min_width and w_i >= min_height)

        if ok:
            kept_images.append(img)
            kept_image_ids.add(int(img["id"]))

    kept_annotations = [
        ann for ann in annotations
        if int(ann.get("image_id", -1)) in kept_image_ids
    ]

    return kept_images, kept_annotations, kept_image_ids, missing_wh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter COCO by minimum image resolution")
    parser.add_argument("--input-json", required=True, help="Input COCO annotation json path")
    parser.add_argument("--image-root", required=True, help="Directory containing source images")
    parser.add_argument("--output-json", required=True, help="Output COCO annotation json path")
    parser.add_argument("--output-image-dir", required=True, help="Directory to copy kept images into")
    parser.add_argument("--min-width", type=int, default=1280, help="Minimum width (default: 1280)")
    parser.add_argument("--min-height", type=int, default=720, help="Minimum height (default: 720)")
    parser.add_argument(
        "--strict-orientation",
        action="store_true",
        help="If set, require width>=min-width and height>=min-height only (no rotated acceptance)",
    )
    parser.add_argument(
        "--drop-unused-categories",
        action="store_true",
        help="Also remove categories not referenced by remaining annotations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json)
    image_root = Path(args.image_root)
    output_json = Path(args.output_json)
    output_image_dir = Path(args.output_image_dir)

    if args.min_width <= 0 or args.min_height <= 0:
        raise ValueError("--min-width and --min-height must be > 0")

    coco = load_json(input_json)
    original_images = coco.get("images", [])
    original_annotations = coco.get("annotations", [])
    original_categories = coco.get("categories", [])

    kept_images, kept_annotations, kept_image_ids, missing_wh = filter_coco_by_resolution(
        coco=coco,
        min_width=args.min_width,
        min_height=args.min_height,
        allow_rotated=not args.strict_orientation,
    )

    new_coco = dict(coco)
    new_coco["images"] = kept_images
    new_coco["annotations"] = kept_annotations

    if args.drop_unused_categories:
        used_cat_ids = {int(ann["category_id"]) for ann in kept_annotations}
        new_coco["categories"] = [
            cat for cat in original_categories
            if int(cat.get("id", -1)) in used_cat_ids
        ]

    copied_count, missing_image_files = copy_kept_images(
        kept_images=kept_images,
        image_root=image_root,
        output_image_dir=output_image_dir,
    )

    save_json(output_json, new_coco)

    print(f"done -> {output_json}")
    print(f"  min resolution:              {args.min_width}x{args.min_height}")
    print(f"  strict orientation:          {args.strict_orientation}")
    print(f"  original images:             {len(original_images)}")
    print(f"  kept images:                 {len(kept_images)}")
    print(f"  dropped images:              {len(original_images) - len(kept_images)}")
    print(f"  images missing width/height: {missing_wh}")
    print(f"  original annotations:        {len(original_annotations)}")
    print(f"  kept annotations:            {len(kept_annotations)}")
    print(f"  dropped annotations:         {len(original_annotations) - len(kept_annotations)}")
    print(f"  copied kept images:          {copied_count}")
    print(f"  missing image files:         {missing_image_files}")
    if args.drop_unused_categories:
        print(f"  categories kept:             {len(new_coco.get('categories', []))} / {len(original_categories)}")
    print(f"  output image dir:            {output_image_dir}")


if __name__ == "__main__":
    main()
