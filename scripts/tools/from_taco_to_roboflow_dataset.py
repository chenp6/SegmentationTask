"""
Merge split-specific TACO COCO annotation files into Roboflow-style split folders.

This script scans annotation files such as:
    annotations_0_train.json
    annotations_1_train.json
    annotations_0_val.json
    annotations_0_test.json

For each split (`train`, `val`, `test`), it:
1. Creates a split directory under the input root if it does not already exist.
2. Merges all matching annotation files for that split.
3. Reassigns image ids and annotation ids to avoid collisions.
4. Writes the merged COCO annotation file to:
   <input-root>/<split>/_annotations.coco.json

Example:
    python -m scripts.tools.from_taco_to_roboflow_dataset \
      --input-root data/taco_dataset
"""

import argparse
import json
from pathlib import Path


def build_supercategory_mapping(categories: list[dict]) -> tuple[list[dict], dict[int, int]]:
    """
    Build new categories using `supercategory` and return old->new category_id mapping.
    """
    super_to_new_id: dict[str, int] = {}
    new_categories: list[dict] = []
    old_to_new: dict[int, int] = {}

    for cat in categories:
        old_id = int(cat["id"])
        super_name = str(cat.get("supercategory") or cat.get("name") or f"category_{old_id}")

        if super_name not in super_to_new_id:
            new_id = len(super_to_new_id) + 1
            super_to_new_id[super_name] = new_id
            new_categories.append(
                {
                    "id": new_id,
                    "name": super_name,
                    "supercategory": super_name,
                }
            )

        old_to_new[old_id] = super_to_new_id[super_name]

    return new_categories, old_to_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Path to TACO Dataset input root path")
    parser.add_argument(
        "--use-supercategory-as-category",
        action="store_true",
        help="Replace fine-grained categories with supercategory labels",
    )
    args = parser.parse_args()
    root = Path(args.input_root)  

    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir =  root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {split_dir}")
        merged = {
            "images": [],
            "annotations": [],
            "categories": None
        }

        ann_id_offset = 0
        img_id_offset = 0

        for json_file in sorted(root.glob(f"annotations_*_{split}.json")):
            print(f"Processing {json_file}")

            data = json.load(open(json_file))

            if merged["categories"] is None:
                if args.use_supercategory_as_category:
                    new_categories, _ = build_supercategory_mapping(data["categories"])
                    merged["categories"] = new_categories
                else:
                    merged["categories"] = data["categories"]

            if args.use_supercategory_as_category:
                _, old_to_new_cat = build_supercategory_mapping(data["categories"])
            else:
                old_to_new_cat = None

            img_id_map = {}

            # images
            for img in data["images"]:
                old_id = img["id"]
                new_id = old_id + img_id_offset
                img_id_map[old_id] = new_id
                img["id"] = new_id

                old_filename = img["file_name"]
                img["file_name"] = "../"+ old_filename
                merged["images"].append(img)

            # annotations
            for ann in data["annotations"]:
                ann["id"] = ann["id"] + ann_id_offset
                ann["image_id"] = img_id_map[ann["image_id"]]
                if old_to_new_cat is not None:
                    ann["category_id"] = old_to_new_cat[int(ann["category_id"])]
                merged["annotations"].append(ann)

            img_id_offset += len(data["images"])
            ann_id_offset += len(data["annotations"])

        out_path = Path(f"{args.input_root}/{split}/_annotations.coco.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        json.dump(merged, open(out_path, "w"))
        print(f"Saved {out_path}")
if __name__ == "__main__":
    main()
