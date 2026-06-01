"""
Merge selected COCO category names into target category names.

Default merge rules:
  - N95 -> mask
  - plastic_medical_bottle -> medical_bottle
  - glass_medical_bottle -> medical_bottle

Usage:
  python -m scripts.tools.merge_coco_classes \
    --input-json data/myset/train/_annotations.coco.json \
    --output-json data/myset/train/_annotations.merged.coco.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path



DEFAULT_MERGE_MAP = {
    "N95": "mask",
    "mask": "mask",
    "plastic_medical_bottle": "medical_bottle",
    "glass_bottle": "medical_bottle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge COCO categories by name")
    parser.add_argument("--input-json", required=True, help="Input COCO annotation json")
    parser.add_argument("--output-json", required=True, help="Output COCO annotation json")
    parser.add_argument(
        "--merge-class",
        action="append",
        default=[],
        help="Optional extra/override rule SRC:DST (repeatable)",
    )
    return parser.parse_args()


def parse_merge_rules(extra_rules: list[str]) -> dict[str, str]:
    merge_map = dict(DEFAULT_MERGE_MAP)
    for rule in extra_rules:
        if ":" not in rule:
            raise ValueError(f"Invalid --merge-class rule: {rule} (expected SRC:DST)")
        src, dst = rule.split(":", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError(f"Invalid --merge-class rule: {rule} (expected SRC:DST)")
        merge_map[src] = dst
    return merge_map


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def merge_categories(coco: dict, merge_map: dict[str, str]) -> dict:
    categories = coco.get("categories", [])
    annotations = coco.get("annotations", [])

    old_id_to_new_name: dict[int, str] = {}
    new_name_to_new_id: dict[str, int] = {}
    merged_categories: list[dict] = []

    # Build merged category table by name.
    for cat in categories:
        old_id = int(cat["id"])
        old_name = str(cat["name"])
        new_name = merge_map.get(old_name, old_name)
        old_id_to_new_name[old_id] = new_name

        if new_name not in new_name_to_new_id:
            new_id = len(new_name_to_new_id) + 1
            new_name_to_new_id[new_name] = new_id
            merged_categories.append(
                {
                    "id": new_id,
                    "name": new_name,
                    "supercategory": cat.get("supercategory", new_name),
                }
            )

    # Remap annotation category ids.
    remapped_annotations = []
    for ann in annotations:
        old_cat_id = int(ann["category_id"])
        if old_cat_id not in old_id_to_new_name:
            continue
        new_name = old_id_to_new_name[old_cat_id]
        new_cat_id = new_name_to_new_id[new_name]
        new_ann = dict(ann)
        new_ann["category_id"] = new_cat_id
        remapped_annotations.append(new_ann)

    out = dict(coco)
    out["categories"] = merged_categories
    out["annotations"] = remapped_annotations
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    coco = load_json(input_path)
    merge_map = parse_merge_rules(args.merge_class)
    merged = merge_categories(coco, merge_map)
    save_json(output_path, merged)

    print("Class merge completed.")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print("Merge rules:")
    for src, dst in merge_map.items():
        print(f"  {src} -> {dst}")
    print(f"Categories: {len(coco.get('categories', []))} -> {len(merged.get('categories', []))}")


if __name__ == "__main__":
    main()

