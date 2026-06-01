"""
Split one COCO dataset into train/valid/test or K-Fold splits.

Input layout (example):
  dataset_root/
    _annotations.coco.json
    images...

Output layout (normal split):
  output_root/
    train/
      _annotations.coco.json
      images... (optional, if --copy-images)
    valid/
      _annotations.coco.json
      images... (optional, if --copy-images)
    test/
      _annotations.coco.json
      images... (optional, if --copy-images)

Output layout (K-Fold):
  output_root/
    fold_0/
      train/_annotations.coco.json
      valid/_annotations.coco.json
      test/_annotations.coco.json
    fold_1/
      ...

Example:
  python -m scripts.tools.split_dataset \
    --dataset-root data/my_dataset \
    --output-root data/my_dataset_split \
    --train-ratio 0.8 \
    --valid-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    --copy-images

  python -m scripts.tools.split_dataset \
    --dataset-root data/my_dataset \
    --output-root data/my_dataset_kfold \
    --k-fold 5 \
    --seed 42 \
    --copy-images
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_ANN_CANDIDATES = (
    "_annotations.coco.json",
    "_annotation.coco.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split one COCO dataset into train/valid/test by image-level random split."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Input dataset root containing one COCO annotation JSON.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output dataset root for split folders.",
    )
    parser.add_argument(
        "--annotation",
        default="",
        help="Optional explicit COCO annotation JSON path.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio. Default: 0.8",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Valid ratio. Default: 0.1",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio. Default: 0.1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to output split folders.",
    )
    parser.add_argument(
        "--k-fold",
        type=int,
        default=0,
        help="Enable K-Fold split. Example: 5 means 5 folds. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=-1,
        help=(
            "When --k-fold is set, export only this fold index (0-based). "
            "Default: -1 means export all folds."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_annotation_path(dataset_root: Path, explicit_path: str) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {p}")
        return p

    for filename in DEFAULT_ANN_CANDIDATES:
        candidate = dataset_root / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Cannot find COCO annotation file under dataset root. "
        f"Checked: {', '.join(DEFAULT_ANN_CANDIDATES)}"
    )


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    for name, value in (
        ("train-ratio", train_ratio),
        ("valid-ratio", valid_ratio),
        ("test-ratio", test_ratio),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    total = train_ratio + valid_ratio + test_ratio
    if total <= 0:
        raise ValueError("Sum of train/valid/test ratios must be > 0")


def resolve_image_path(dataset_root: Path, file_name: str) -> Path | None:
    direct = dataset_root / file_name
    if direct.exists():
        return direct

    by_name = dataset_root / Path(file_name).name
    if by_name.exists():
        return by_name

    target = Path(file_name).name
    for p in dataset_root.rglob(target):
        if p.is_file() and p.name == target:
            return p

    return None


def calculate_split_sizes(total: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    train_ratio, valid_ratio, test_ratio = ratios
    ratio_sum = train_ratio + valid_ratio + test_ratio

    raw = [
        total * train_ratio / ratio_sum,
        total * valid_ratio / ratio_sum,
        total * test_ratio / ratio_sum,
    ]
    base = [int(x) for x in raw]
    remain = total - sum(base)

    frac = [(i, raw[i] - base[i]) for i in range(3)]
    frac.sort(key=lambda x: x[1], reverse=True)

    for i in range(remain):
        base[frac[i % 3][0]] += 1

    return base[0], base[1], base[2]


def calculate_fold_sizes(total: int, k_fold: int) -> List[int]:
    base = total // k_fold
    remain = total % k_fold
    sizes = [base] * k_fold
    for i in range(remain):
        sizes[i] += 1
    return sizes


def get_fold_ranges(total: int, k_fold: int) -> List[Tuple[int, int]]:
    sizes = calculate_fold_sizes(total, k_fold)
    ranges: List[Tuple[int, int]] = []
    start = 0
    for size in sizes:
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def build_split_coco(
    split_images: List[dict],
    anns_by_img: Dict[int, List[dict]],
    categories: List[dict],
    coco: dict,
) -> dict:
    out_images: List[dict] = []
    out_annotations: List[dict] = []
    next_img_id = 1
    next_ann_id = 1
    id_map: Dict[int, int] = {}

    for img in split_images:
        old_img_id = int(img["id"])
        new_img_id = next_img_id
        next_img_id += 1
        id_map[old_img_id] = new_img_id

        img_out = dict(img)
        img_out["id"] = new_img_id
        img_out["file_name"] = Path(str(img["file_name"])).name
        out_images.append(img_out)

    for old_img_id, new_img_id in id_map.items():
        for ann in anns_by_img.get(old_img_id, []):
            ann_out = dict(ann)
            ann_out["id"] = next_ann_id
            next_ann_id += 1
            ann_out["image_id"] = new_img_id
            out_annotations.append(ann_out)

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories,
    }


def maybe_copy_images(
    split_images: List[dict],
    dataset_root: Path,
    split_dir: Path,
    copy_images: bool,
) -> None:
    if not copy_images:
        return

    for img in split_images:
        out_file_name = Path(str(img["file_name"])).name
        src_img = resolve_image_path(dataset_root, str(img["file_name"]))
        if src_img is None:
            raise FileNotFoundError(
                f"Image not found for file_name='{img['file_name']}' under {dataset_root}"
            )
        shutil.copy2(src_img, split_dir / out_file_name)


def split_dataset(
    coco: dict,
    dataset_root: Path,
    output_root: Path,
    ratios: Tuple[float, float, float],
    seed: int,
    copy_images: bool,
) -> None:
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))
    categories = list(coco.get("categories", []))

    if not images:
        raise ValueError("No images found in COCO file.")

    anns_by_img: Dict[int, List[dict]] = {}
    for ann in annotations:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    rng = random.Random(seed)
    rng.shuffle(images)

    n_train, n_valid, n_test = calculate_split_sizes(len(images), ratios)
    train_imgs = images[:n_train]
    valid_imgs = images[n_train:n_train + n_valid]
    test_imgs = images[n_train + n_valid:n_train + n_valid + n_test]

    split_items = [("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs)]

    for split_name, split_images in split_items:
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        maybe_copy_images(split_images, dataset_root, split_dir, copy_images)
        out_coco = build_split_coco(split_images, anns_by_img, categories, coco)

        save_json(split_dir / "_annotations.coco.json", out_coco)

        print(
            f"[{split_name}] images={len(out_coco['images'])}, annotations={len(out_coco['annotations'])} "
            f"-> {(split_dir / '_annotations.coco.json').as_posix()}"
        )


def split_dataset_kfold(
    coco: dict,
    dataset_root: Path,
    output_root: Path,
    k_fold: int,
    valid_ratio: float,
    seed: int,
    copy_images: bool,
    fold_index: int,
) -> None:
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))
    categories = list(coco.get("categories", []))

    if not images:
        raise ValueError("No images found in COCO file.")
    if k_fold < 2:
        raise ValueError("--k-fold must be >= 2")
    if k_fold > len(images):
        raise ValueError(f"--k-fold ({k_fold}) cannot exceed number of images ({len(images)})")
    if fold_index >= k_fold:
        raise ValueError(f"--fold-index must be < --k-fold ({k_fold})")
    if valid_ratio < 0:
        raise ValueError("--valid-ratio must be >= 0")

    anns_by_img: Dict[int, List[dict]] = {}
    for ann in annotations:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    rng = random.Random(seed)
    rng.shuffle(images)
    fold_ranges = get_fold_ranges(len(images), k_fold)

    fold_indices = range(k_fold) if fold_index < 0 else [fold_index]

    for i in fold_indices:
        test_start, test_end = fold_ranges[i]
        test_imgs = images[test_start:test_end]
        train_pool = images[:test_start] + images[test_end:]

        n_valid = int(round(len(train_pool) * valid_ratio))
        n_valid = min(max(n_valid, 0), len(train_pool))
        valid_imgs = train_pool[:n_valid]
        train_imgs = train_pool[n_valid:]

        fold_root = output_root / f"fold_{i}"
        split_items = [("train", train_imgs), ("valid", valid_imgs), ("test", test_imgs)]

        for split_name, split_images in split_items:
            split_dir = fold_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            maybe_copy_images(split_images, dataset_root, split_dir, copy_images)
            out_coco = build_split_coco(split_images, anns_by_img, categories, coco)
            save_json(split_dir / "_annotations.coco.json", out_coco)
            print(
                f"[fold {i}][{split_name}] images={len(out_coco['images'])}, "
                f"annotations={len(out_coco['annotations'])} "
                f"-> {(split_dir / '_annotations.coco.json').as_posix()}"
            )


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    ann_path = resolve_annotation_path(dataset_root, args.annotation)
    coco = load_json(ann_path)

    if args.k_fold > 0:
        split_dataset_kfold(
            coco=coco,
            dataset_root=dataset_root,
            output_root=output_root,
            k_fold=args.k_fold,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
            copy_images=args.copy_images,
            fold_index=args.fold_index,
        )
    else:
        split_dataset(
            coco=coco,
            dataset_root=dataset_root,
            output_root=output_root,
            ratios=(args.train_ratio, args.valid_ratio, args.test_ratio),
            seed=args.seed,
            copy_images=args.copy_images,
        )

    print("Done.")


if __name__ == "__main__":
    main()
