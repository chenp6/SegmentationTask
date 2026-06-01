"""
Usage:
    python -m scripts.tools.select_sharpest_every_n \
        --input-root data/grouped_images \
        --output-root data/grouped_images_best

Example(copy):
    python -m scripts.tools.select_sharpest_every_n \
        --input-root data/ward_video/multi_perspectives  \
        --output-root data/ward_video/multi_perspectives_best_new2 \
        --group-size 60 \
        --workers 8
Example(dry-run):
    python -m scripts.tools.select_sharpest_every_n \
        --input-root data/ward_video/multi_perspectives  \
        --output-root data/ward_video/multi_perspectives_best \
        --group-size 60 \
        --workers 8 \
        --dry-run
Description:
    For each tag folder under <input-root>, sort image files by file name,
    split them into fixed-size groups, and keep only the sharpest image from
    each group.

    If the last group has fewer than N images, one sharpest image is still
    selected from that smaller group.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each tag folder, sort images by file name and keep the "
            "sharpest image from every fixed-size group"
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory that contains per-tag folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for selected images",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=30,
        help="Number of images per group. Default: 30",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files without writing output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of worker threads for sharpness scoring. "
            "Default: auto (CPU count)"
        ),
    )
    return parser.parse_args()


def iter_tag_dirs(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.iterdir() if path.is_dir())


def iter_image_files(tag_dir: Path) -> list[Path]:
    # Sort by file name as requested.
    # 依照檔名排序。
    return sorted(
        path
        for path in tag_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def chunked(items: list[Path], group_size: int) -> list[list[Path]]:
    return [items[index: index + group_size] for index in range(0, len(items), group_size)]


def sharpness_score(image_path: Path) -> float:
    # Use Laplacian variance as a simple and common sharpness metric.
    # 使用 Laplacian variance 作為常見的清晰度分數。
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def compute_scores_parallel(
    image_files: list[Path],
    workers: int,
) -> dict[Path, float]:
    if not image_files:
        return {}

    max_workers = workers if workers > 0 else (os.cpu_count() or 1)
    scores: dict[Path, float] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(sharpness_score, path): path for path in image_files}
        for future in as_completed(futures):
            path = futures[future]
            scores[path] = future.result()

    return scores


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)



def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if args.group_size < 1:
        raise ValueError("--group-size must be >= 1")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")

    tag_dirs = iter_tag_dirs(input_root)
    if not tag_dirs:
        print(f"No tag directories found under: {input_root}")
        return

    total_selected = 0
    for tag_dir in tag_dirs:
        image_files = iter_image_files(tag_dir)
        if not image_files:
            print(f"Skipping {tag_dir.name}: no image files found")
            continue

        scores = compute_scores_parallel(image_files, args.workers)
        groups = chunked(image_files, args.group_size)
        selected_for_tag = 0

        for group_index, group_files in enumerate(groups, start=1):
            best_path = max(group_files, key=scores.__getitem__)
            dst_path = output_root / tag_dir.name / best_path.name

            print(
                f"[{tag_dir.name}] group {group_index}: "
                f"selected {best_path.name} from {len(group_files)} image(s)"
            )

            if not args.dry_run:
                copy_file(best_path, dst_path)

            selected_for_tag += 1
            total_selected += 1

        print(
            f"[{tag_dir.name}] selected {selected_for_tag} image(s) "
            f"from {len(image_files)} total image(s)"
        )

    if args.dry_run:
        print(f"Dry run complete. Planned {total_selected} selected image(s).")
    else:
        print(f"Selection complete. Wrote {total_selected} image(s) to: {output_root}")


if __name__ == "__main__":
    main()
