"""
Merge two COCO datasets by simple concatenation and global category reindexing.

What this script does:
1) Merge dataset A and B images/annotations for each split (no IoU filtering, no priority drop).
2) Build ONE global category mapping from ALL requested splits across A+B.
3) Reindex category_id with that single mapping so every output split uses identical categories.
4) Reindex image_id / annotation id in output files.

Input layout (Roboflow-style):
  dataset_a_root/
    train/_annotations.coco.json
    valid/_annotations.coco.json
    test/_annotations.coco.json
  dataset_b_root/
    train/_annotations.coco.json
    valid/_annotations.coco.json
    test/_annotations.coco.json

Output layout:
  output_root/
    <split>/
      images... (optional, if --copy-images)
      _annotations.coco.json

Example:
  python -m scripts.tools.merge_coco_datasets_by_priority \
    --dataset-a-root data/A_dataset \
    --dataset-b-root data/B_dataset \
    --output-root data/merged_dataset \
    --splits train,valid,test \
    --copy-images
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import re
from typing import Dict, List, Tuple


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _canonicalize_category_key(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def normalize_category_name(name: str) -> str:
    # Keep compatibility with known naming variants in existing pipelines.
    alias_map = {
        "solid_linen_bin": "soiled_linen_bin",
        "light_switcher": "light_switch",
        "bedcurtain": "bed_curtain",
    }
    key = _canonicalize_category_key(name)
    return alias_map.get(key, key)


def build_global_category_mapping(
    dataset_a_root: Path,
    dataset_b_root: Path,
    splits: List[str],
) -> Tuple[List[dict], Dict[int, int], Dict[int, int]]:
    """
    Build one global category mapping across all splits and both datasets.

    Returns:
      categories_out: list of {id, name, supercategory}
      map_a: old category_id in A -> new category_id
      map_b: old category_id in B -> new category_id
    """
    key_to_new_id: Dict[str, int] = {}
    categories_out: List[dict] = []
    map_a: Dict[int, int] = {}
    map_b: Dict[int, int] = {}

    def ensure_category(name: str, supercategory: str | None) -> int:
        normalized = normalize_category_name(str(name))
        if normalized in key_to_new_id:
            return key_to_new_id[normalized]

        new_id = len(key_to_new_id) + 1
        key_to_new_id[normalized] = new_id
        categories_out.append(
            {
                "id": new_id,
                "name": normalized,
                "supercategory": supercategory or normalized,
            }
        )
        return new_id

    for split in splits:
        a_ann = dataset_a_root / split / "_annotations.coco.json"
        b_ann = dataset_b_root / split / "_annotations.coco.json"

        if a_ann.exists():
            coco_a = load_json(a_ann)
            for c in coco_a.get("categories", []):
                new_id = ensure_category(c["name"], c.get("supercategory"))
                map_a[int(c["id"])] = new_id

        if b_ann.exists():
            coco_b = load_json(b_ann)
            for c in coco_b.get("categories", []):
                new_id = ensure_category(c["name"], c.get("supercategory"))
                map_b[int(c["id"])] = new_id

    return categories_out, map_a, map_b


def resolve_image_path(split_dir: Path, file_name: str) -> Path | None:
    p1 = split_dir / file_name
    if p1.exists():
        return p1

    p2 = split_dir / Path(file_name).name
    if p2.exists():
        return p2

    target = Path(file_name).name
    for p in split_dir.iterdir():
        if p.is_file() and p.name == target:
            return p

    return None


def merge_split(
    split: str,
    dataset_a_root: Path,
    dataset_b_root: Path,
    output_root: Path,
    categories_out: List[dict],
    map_a: Dict[int, int],
    map_b: Dict[int, int],
    copy_images: bool,
) -> None:
    a_split_dir = dataset_a_root / split
    b_split_dir = dataset_b_root / split
    out_split_dir = output_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    a_ann_path = a_split_dir / "_annotations.coco.json"
    b_ann_path = b_split_dir / "_annotations.coco.json"

    if not a_ann_path.exists() and not b_ann_path.exists():
        print(f"[skip] {split}: no annotation in A/B")
        return

    coco_a = load_json(a_ann_path) if a_ann_path.exists() else {"images": [], "annotations": [], "categories": []}
    coco_b = load_json(b_ann_path) if b_ann_path.exists() else {"images": [], "annotations": [], "categories": []}

    out_images: List[dict] = []
    out_annotations: List[dict] = []

    next_img_id = 1
    next_ann_id = 1

    def append_dataset(coco: dict, cat_map: Dict[int, int], src_split_dir: Path) -> None:
        nonlocal next_img_id, next_ann_id

        anns_by_img: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

        for img in coco.get("images", []):
            old_img_id = int(img["id"])
            new_img_id = next_img_id
            next_img_id += 1

            file_name = Path(str(img["file_name"])).name
            out_images.append(
                {
                    "id": new_img_id,
                    "file_name": file_name,
                    "width": int(img["width"]),
                    "height": int(img["height"]),
                }
            )

            if copy_images:
                src_img = resolve_image_path(src_split_dir, str(img["file_name"]))
                if src_img is not None:
                    shutil.copy2(src_img, out_split_dir / file_name)

            for ann in anns_by_img.get(old_img_id, []):
                old_cat_id = int(ann["category_id"])
                if old_cat_id not in cat_map:
                    raise KeyError(
                        f"Category id {old_cat_id} not found in mapping for split '{split}'."
                    )

                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                next_ann_id += 1
                new_ann["image_id"] = new_img_id
                new_ann["category_id"] = cat_map[old_cat_id]
                out_annotations.append(new_ann)

    append_dataset(coco_a, map_a, a_split_dir)
    append_dataset(coco_b, map_b, b_split_dir)

    out_coco = {
        "info": coco_a.get("info", coco_b.get("info", {})),
        "licenses": coco_a.get("licenses", coco_b.get("licenses", [])),
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories_out,
    }

    save_json(out_split_dir / "_annotations.coco.json", out_coco)
    print(f"[{split}] images={len(out_images)} anns={len(out_annotations)} cats={len(categories_out)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two COCO datasets and reindex category_id globally across all splits"
    )
    parser.add_argument("--dataset-a-root", required=True, help="Dataset A root")
    parser.add_argument("--dataset-b-root", required=True, help="Dataset B root")
    parser.add_argument("--output-root", required=True, help="Output merged dataset root")
    parser.add_argument("--splits", default="train,valid,test", help="Comma-separated splits")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output split folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_a_root = Path(args.dataset_a_root).resolve()
    dataset_b_root = Path(args.dataset_b_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    categories_out, map_a, map_b = build_global_category_mapping(
        dataset_a_root=dataset_a_root,
        dataset_b_root=dataset_b_root,
        splits=splits,
    )

    if not categories_out:
        raise RuntimeError("No categories found from the given inputs.")

    for split in splits:
        merge_split(
            split=split,
            dataset_a_root=dataset_a_root,
            dataset_b_root=dataset_b_root,
            output_root=output_root,
            categories_out=categories_out,
            map_a=map_a,
            map_b=map_b,
            copy_images=bool(args.copy_images),
        )

    print(f"\nDone. Output: {output_root}")


if __name__ == "__main__":
    main()
