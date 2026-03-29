"""
將固定格式的 COCO instance 資料轉成 Roboflow 風格的 split annotation 輸出。
Convert COCO instance data in a fixed input layout into Roboflow-style
split annotation outputs.

輸入格式 Input layout:
    <input-root>/rgb_image/<tag>/*.png
    <input-root>/coco_json/<tag>.json

這裡的 tag 只是資料夾名稱與 json 檔名的對應標記，不代表類別名稱。
Here, tag is only a pairing label for the folder name and json filename,
not a category name.

輸出格式 Output layout:
    <input-root>/train/_annotations.coco.json
    <input-root>/valid/_annotations.coco.json
    <input-root>/test/_annotations.coco.json

流程說明 Pipeline:
1. 掃描 <input-root>/coco_json/*.json
2. 合併所有 COCO json 成單一資料集
3. 以 image 為單位隨機切分為 train、valid、test
4. 只輸出 split annotation 檔案，不複製原始圖片
5. annotation 中的 file_name 直接指向原始 rgb_image/<tag>/ 路徑


Usage:
    python -m scripts.tools.from_ward_to_roboflow_dataset --input-root data/ward_dataset
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    """解析命令列參數。 Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "將固定輸入格式的 COCO instance 資料轉成 split annotation 輸出。 "
            "Convert COCO instance data in the fixed input layout into "
            "split annotation outputs."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help=(
            "輸入根目錄，需包含 rgb_image 與 coco_json。 "
            "Input root containing rgb_image and coco_json."
        ),
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALID", "TEST"),
        default=(0.7, 0.15, 0.15),
        help=(
            "train/valid/test 的切分比例。 "
            "Split ratios for train/valid/test."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "隨機切分使用的亂數種子。 "
            "Random seed used for image splitting."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """讀取 JSON 檔案。 Load a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """寫入 JSON 檔案。 Save data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def normalize_ratios(ratios: Sequence[float]) -> Tuple[float, float, float]:
    """正規化切分比例並驗證輸入。 Normalize split ratios and validate input."""
    if len(ratios) != 3:
        raise ValueError("--split-ratios must contain exactly 3 values.")
    if any(r < 0 for r in ratios):
        raise ValueError("--split-ratios cannot contain negative values.")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("--split-ratios must sum to a positive value.")

    return tuple(r / total for r in ratios)


def collect_coco_json_files(input_root: Path) -> list[Path]:
    """收集 coco_json 目錄下的標註檔。 Collect annotation files under coco_json."""
    coco_json_dir = input_root / "coco_json"
    if not coco_json_dir.is_dir():
        raise FileNotFoundError(f"COCO json directory not found: {coco_json_dir}")

    json_files = sorted(path for path in coco_json_dir.glob("*.json") if path.is_file())
    if not json_files:
        raise FileNotFoundError(f"No json files found under {coco_json_dir}")

    return json_files


def merge_coco_files(json_files: Iterable[Path], input_root: Path) -> dict:
    """
    合併多個 COCO json，並重新編排 image/annotation id。
    Merge multiple COCO json files and reindex image/annotation ids.
    """
    merged = {"images": [], "annotations": [], "categories": []}
    categories_by_id: Dict[int, dict] = {}
    next_image_id = 1
    next_annotation_id = 1

    for json_file in json_files:
        print(f"Processing {json_file}")
        data = load_json(json_file)
        tag = json_file.stem
        image_id_map: Dict[int, int] = {}

        # 依 category id 合併 categories
        # Merge categories by category id.
        for category in data.get("categories", []):
            current_category_id = category["id"]
            if current_category_id not in categories_by_id:
                categories_by_id[current_category_id] = copy.deepcopy(category)

        # 重新指定 image id，並把 file_name 改成指向原始 rgb_image/<tag>/ 路徑
        # Reassign image ids and rewrite file_name to point at the original rgb_image/<tag>/ path.
        for image in data.get("images", []):
            new_image = copy.deepcopy(image)
            image_id_map[image["id"]] = next_image_id
            new_image["id"] = next_image_id

            original_file_name = Path(image["file_name"]).name
            new_image["file_name"] = f"../rgb_image/{tag}/{original_file_name}"
            merged["images"].append(new_image)
            next_image_id += 1

        # 重新指定 annotation id，並同步更新 image_id
        # Reassign annotation ids and remap image_id references.
        for annotation in data.get("annotations", []):
            source_image_id = annotation["image_id"]
            if source_image_id not in image_id_map:
                continue

            new_annotation = copy.deepcopy(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = image_id_map[source_image_id]
            merged["annotations"].append(new_annotation)
            next_annotation_id += 1

    merged["categories"] = [
        categories_by_id[key] for key in sorted(categories_by_id.keys())
    ]
    return merged


def assign_random_splits(
    images: Sequence[dict], ratios: Sequence[float], seed: int
) -> Dict[int, str]:
    """依 image 隨機分配 split。 Assign each image to a split at random."""
    train_ratio, valid_ratio, _ = normalize_ratios(ratios)
    image_ids = [image["id"] for image in images]

    rng = random.Random(seed)
    rng.shuffle(image_ids)

    total = len(image_ids)
    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)

    assignments: Dict[int, str] = {}
    for image_id in image_ids[:n_train]:
        assignments[image_id] = "train"
    for image_id in image_ids[n_train : n_train + n_valid]:
        assignments[image_id] = "valid"
    for image_id in image_ids[n_train + n_valid :]:
        assignments[image_id] = "test"

    return assignments


def build_split_dataset(dataset: dict, image_ids: Iterable[int]) -> dict:
    """
    根據指定 image id 建立單一 split 的 COCO 資料。
    Build a single split dataset from selected image ids.
    """
    selected_image_ids = set(image_ids)
    image_id_map: Dict[int, int] = {}
    annotation_id = 1

    split_dataset = {"images": [], "annotations": [], "categories": []}
    used_category_ids = set()

    for image in dataset["images"]:
        if image["id"] not in selected_image_ids:
            continue

        # 每個 split 重新建立連續 image id
        # Reindex image ids sequentially within each split.
        new_image = copy.deepcopy(image)
        new_image_id = len(split_dataset["images"]) + 1
        image_id_map[image["id"]] = new_image_id
        new_image["id"] = new_image_id
        split_dataset["images"].append(new_image)

    for annotation in dataset["annotations"]:
        old_image_id = annotation["image_id"]
        if old_image_id not in image_id_map:
            continue

        # annotation id 與 image_id 一併重建
        # Rebuild annotation ids and remap image_id together.
        new_annotation = copy.deepcopy(annotation)
        new_annotation["id"] = annotation_id
        new_annotation["image_id"] = image_id_map[old_image_id]
        split_dataset["annotations"].append(new_annotation)
        used_category_ids.add(new_annotation["category_id"])
        annotation_id += 1

    split_dataset["categories"] = [
        copy.deepcopy(category)
        for category in dataset["categories"]
        if category["id"] in used_category_ids
    ]
    return split_dataset


def write_split_dataset(input_root: Path, split: str, dataset: dict) -> None:
    """寫出單一 split 的 annotation 檔案。 Write the annotation file for one split."""
    split_dir = input_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    out_path = split_dir / "_annotations.coco.json"
    save_json(out_path, dataset)
    print(
        f"Saved {out_path} "
        f"(images={len(dataset['images'])}, annotations={len(dataset['annotations'])})"
    )


def run_random_split(input_root: Path, args: argparse.Namespace) -> None:
    """執行合併與隨機切分流程。 Run the merge-and-random-split pipeline."""
    json_files = collect_coco_json_files(input_root)
    merged = merge_coco_files(json_files, input_root)
    if not merged["images"]:
        raise ValueError("No images were found in the input COCO files.")

    split_by_image_id = assign_random_splits(
        merged["images"], args.split_ratios, args.seed
    )

    for split in SPLITS:
        # 收集目前 split 對應的 image id，再建立輸出資料
        # Collect image ids for the current split, then build the output dataset.
        image_ids = [
            image["id"]
            for image in merged["images"]
            if split_by_image_id.get(image["id"]) == split
        ]
        split_dataset = build_split_dataset(merged, image_ids)
        write_split_dataset(input_root, split, split_dataset)


def main() -> None:
    """主程式入口。 Program entry point."""
    args = parse_args()
    input_root = Path(args.input_root)
    run_random_split(input_root, args)


if __name__ == "__main__":
    main()
