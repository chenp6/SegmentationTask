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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, help="Path to TACO Dataset input root path")
    args = parser.parse_args()
    root = Path(args.input_root)  

    splits = ["train", "val", "test"]

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
                merged["categories"] = data["categories"]

            img_id_map = {}

            # images
            for img in data["images"]:
                old_id = img["id"]
                new_id = old_id + img_id_offset
                img_id_map[old_id] = new_id
                img["id"] = new_id

                old_filename = img["file_name"]
                img["file_name"] = "../"+old_filename
                merged["images"].append(img)

            # annotations
            for ann in data["annotations"]:
                ann["id"] = ann["id"] + ann_id_offset
                ann["image_id"] = img_id_map[ann["image_id"]]
                merged["annotations"].append(ann)

            img_id_offset += len(data["images"])
            ann_id_offset += len(data["annotations"])

        out_path = Path(f"{args.input_root}/{split}/_annotations.coco.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        json.dump(merged, open(out_path, "w"))
        print(f"Saved {out_path}")
if __name__ == "__main__":
    main()
