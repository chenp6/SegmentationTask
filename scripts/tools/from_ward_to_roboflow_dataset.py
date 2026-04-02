"""
Convert COCO instance data in a fixed input layout into Roboflow-style split annotation outputs.

Input layout:
    <input-root>/rgb_image/<tag>_rgb/*.png
    <input-root>/coco_json/<tag>.json

Output layout:
    <input-root>/train/_annotations.coco.json
    <input-root>/valid/_annotations.coco.json
    <input-root>/test/_annotations.coco.json

Optional preview layout:
    <input-root>/ground_truth_preview/<split>/*.jpg

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

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from pycocotools import mask as coco_mask

SPLITS = ("train", "valid", "test")
EXCLUDED_CATEGORY_IDS = {998, 999}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
    parser.add_argument(
        "--export-ground-truth-images",
        action="store_true",
        help=(
            "額外輸出 ground truth 視覺化影像。"
            "Also export ground-truth visualization images."
        ),
    )
    parser.add_argument(
        "--ground-truth-output-dir",
        default=None,
        help=(
            "ground truth 視覺化影像輸出目錄，預設為 <input-root>/ground_truth_preview。"
            "Directory for ground-truth visualization images. "
            "Default: <input-root>/ground_truth_preview."
        ),
    )
    parser.add_argument(
        "--max-preview-images",
        type=int,
        default=None,
        help=(
            "每個 split 最多輸出幾張 ground truth 視覺化影像。"
            "Maximum number of preview images to export per split."
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


def anns_by_image_id(annotations: Iterable[dict]) -> Dict[int, list[dict]]:
    """依 image_id 分組 annotation。Group annotations by image_id."""
    grouped: Dict[int, list[dict]] = {}
    for annotation in annotations:
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    return grouped


def segmentation_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    """將 COCO polygon/RLE segmentation 轉成 binary mask。Convert COCO polygon/RLE segmentation into a binary mask."""
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        return np.zeros((height, width), dtype=np.uint8)

    decoded = coco_mask.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def color_for_category(category_id: int) -> tuple[int, int, int]:
    """為 category 產生穩定顏色。Generate a stable RGB color for a category."""
    palette = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
        "#808000",
        "#ffd8b1",
        "#000075",
        "#808080",
    ]
    return ImageColor.getrgb(palette[category_id % len(palette)])


def resolve_split_image_path(input_root: Path, split: str, file_name: str) -> Path:
    """從 split annotation 中的 file_name 找回原始影像。Resolve the real image path from a split annotation file_name."""
    return (input_root / split / file_name).resolve()


def merge_coco_files(json_files: Iterable[Path], input_root: Path) -> dict:
    """Merge multiple COCO json files and reindex image/annotation/category ids."""
    merged = {"images": [], "annotations": [], "categories": []}
    temp_annotations: list[dict] = []

    categories_by_id: Dict[int, dict] = {}
    next_image_id = 1
    next_annotation_id = 1

    for json_file in json_files:
        print(f"Processing {json_file}")
        data = load_json(json_file)
        tag = json_file.stem
        image_id_map: Dict[int, int] = {}

        # 依 category id 合併 categories，並排除不需要的 998/999。
        # Merge categories by category id and exclude unwanted 998/999.
        for category in data.get("categories", []):
            current_category_id = int(category["id"])
            if (
                current_category_id in categories_by_id
                or current_category_id in EXCLUDED_CATEGORY_IDS
            ):
                continue
            categories_by_id[current_category_id] = copy.deepcopy(category)

        # 重新指定 image id，並把 file_name 改成指向原始 rgb_image/<tag>_rgb/ 路徑。
        # Reassign image ids and rewrite file_name to point at the original rgb_image/<tag>_rgb/ path.
        for image in data.get("images", []):
            new_image = copy.deepcopy(image)
            original_file_name = Path(image["file_name"]).name
            parts = original_file_name.split("_", 1)
            if len(parts) == 2:
                new_file_name = f"rgb_{parts[1]}"
            else:
                new_file_name = f"rgb_{original_file_name}"

            new_image["file_name"] = f"../rgb_image/{tag}_rgb/{new_file_name}"
            file_path = input_root / "rgb_image" / f"{tag}_rgb" / new_file_name
            if not file_path.is_file():
                continue

            image_id_map[int(image["id"])] = next_image_id
            new_image["id"] = next_image_id
            merged["images"].append(new_image)
            next_image_id += 1

        # 重新指定 annotation id，並暫存舊 category_id 以便後續 remap。
        # Reassign annotation ids and keep old category_id for later remapping.
        for annotation in data.get("annotations", []):
            old_category_id = int(annotation["category_id"])
            if old_category_id in EXCLUDED_CATEGORY_IDS:
                continue

            source_image_id = int(annotation["image_id"])
            if source_image_id not in image_id_map:
                continue

            new_annotation = copy.deepcopy(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = image_id_map[source_image_id]
            temp_annotations.append(new_annotation)
            next_annotation_id += 1

    # 依舊 category id 排序後重新編連續的新 category id。
    # Rebuild contiguous category ids from sorted old category ids.
    new_categories_id_map: Dict[int, int] = {}
    for index, old_id in enumerate(sorted(categories_by_id), start=1):
        new_category = copy.deepcopy(categories_by_id[old_id])
        new_category["id"] = index
        merged["categories"].append(new_category)
        new_categories_id_map[old_id] = index

    # 用 old_category_id -> new_category_id 對映更新 annotations。
    # Update annotations with the old_category_id -> new_category_id mapping.
    for annotation in temp_annotations:
        remapped_annotation = copy.deepcopy(annotation)
        remapped_annotation["category_id"] = new_categories_id_map[
            int(remapped_annotation["category_id"])
        ]
        merged["annotations"].append(remapped_annotation)

    return merged


def assign_random_splits(
    images: Sequence[dict], ratios: Sequence[float], seed: int
) -> Dict[int, str]:
    """Assign each image to a split at random."""
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
    """Build a single split dataset from selected image ids."""
    selected_image_ids = set(image_ids)
    image_id_map: Dict[int, int] = {}
    annotation_id = 1

    split_dataset = {"images": [], "annotations": [], "categories": []}
    used_category_ids = set()

    for image in dataset["images"]:
        if image["id"] not in selected_image_ids:
            continue

        new_image = copy.deepcopy(image)
        new_image_id = len(split_dataset["images"]) + 1
        image_id_map[int(image["id"])] = new_image_id
        new_image["id"] = new_image_id
        split_dataset["images"].append(new_image)

    for annotation in dataset["annotations"]:
        old_image_id = int(annotation["image_id"])
        if old_image_id not in image_id_map:
            continue

        new_annotation = copy.deepcopy(annotation)
        new_annotation["id"] = annotation_id
        new_annotation["image_id"] = image_id_map[old_image_id]
        split_dataset["annotations"].append(new_annotation)
        used_category_ids.add(int(new_annotation["category_id"]))
        annotation_id += 1

    split_dataset["categories"] = [
        copy.deepcopy(category)
        for category in dataset["categories"]
        if int(category["id"]) in used_category_ids
    ]
    return split_dataset


def write_split_dataset(input_root: Path, split: str, dataset: dict) -> None:
    """Write the annotation file for one split."""
    split_dir = input_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    out_path = split_dir / "_annotations.coco.json"
    save_json(out_path, dataset)
    print(
        f"Saved {out_path} "
        f"(images={len(dataset['images'])}, annotations={len(dataset['annotations'])})"
    )


def export_ground_truth_images(
    input_root: Path,
    split: str,
    dataset: dict,
    output_dir: Path,
    max_preview_images: int | None,
) -> None:
    """輸出 ground truth 視覺化影像。Export ground-truth visualization images."""
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    annotations_by_image = anns_by_image_id(dataset.get("annotations", []))
    categories_by_id = {
        int(category["id"]): category for category in dataset.get("categories", [])
    }

    image_count = 0
    for image_info in dataset.get("images", []):
        if max_preview_images is not None and image_count >= max_preview_images:
            break

        image_path = resolve_split_image_path(
            input_root=input_root,
            split=split,
            file_name=image_info["file_name"],
        )
        if not image_path.exists():
            print(f"Skipping preview image because source was not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for annotation in annotations_by_image.get(int(image_info["id"]), []):
            category_id = int(annotation["category_id"])
            category = categories_by_id.get(category_id, {})
            category_name = str(category.get("name", category_id))
            color = color_for_category(category_id)
            fill_color = (color[0], color[1], color[2], 96)

            mask = segmentation_to_binary_mask(
                annotation.get("segmentation"),
                height=int(image_info["height"]),
                width=int(image_info["width"]),
            )
            if mask.any():
                mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
                color_layer = Image.new("RGBA", image.size, fill_color)
                overlay.paste(color_layer, (0, 0), mask_image)

            bbox = annotation.get("bbox")
            if bbox and len(bbox) >= 4:
                x, y, w, h = [float(v) for v in bbox[:4]]
                draw.rectangle(
                    [(x, y), (x + w, y + h)],
                    outline=color,
                    width=2,
                )
                draw.text((x + 2, y + 2), category_name, fill=color)

        composed = Image.alpha_composite(image, overlay).convert("RGB")
        output_path = split_output_dir / f"{Path(image_info['id']).stem}__gt.jpg"
        composed.save(output_path, quality=95)
        image_count += 1

    print(f"Saved ground-truth preview images to: {split_output_dir}")


def run_random_split(input_root: Path, args: argparse.Namespace) -> None:
    """Run the merge-and-random-split pipeline."""
    json_files = collect_coco_json_files(input_root)
    merged = merge_coco_files(json_files, input_root)
    if not merged["images"]:
        raise ValueError("No images were found in the input COCO files.")

    split_by_image_id = assign_random_splits(
        merged["images"], args.split_ratios, args.seed
    )

    ground_truth_output_dir = (
        Path(args.ground_truth_output_dir)
        if args.ground_truth_output_dir
        else input_root / "ground_truth_preview"
    )

    for split in SPLITS:
        image_ids = [
            image["id"]
            for image in merged["images"]
            if split_by_image_id.get(image["id"]) == split
        ]
        split_dataset = build_split_dataset(merged, image_ids)
        write_split_dataset(input_root, split, split_dataset)

        if args.export_ground_truth_images:
            export_ground_truth_images(
                input_root=input_root,
                split=split,
                dataset=split_dataset,
                output_dir=ground_truth_output_dir,
                max_preview_images=args.max_preview_images,
            )


def main() -> None:
    """Program entry point."""
    args = parse_args()
    input_root = Path(args.input_root)
    run_random_split(input_root, args)


if __name__ == "__main__":
    main()
