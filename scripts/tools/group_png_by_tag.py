"""
Usage:
    python -m scripts.tools.group_png_by_tag \
        --input-root <input-root> \
        --output-root <output-root>

Example(usage):
    python -m scripts.tools.group_png_by_tag \
        --input-root data/ward_video \
        --output-root data/ward_video/multi_perspectives \
        --tag-level 1

Example(dry-run):
    python -m scripts.tools.group_png_by_tag \
    --input-root /data/ward_video \
    --output-root data/ward_video/multi_perspectives \
    --tag-level 1 \
    --dry-run

Description:
    Collect PNG files under <input-root> and copy them into
    <output-root>/<tag>/.

    The tag is resolved from the file's ancestor folders:
    - tag-level 1: use the parent folder name
    - tag-level 2: use the grandparent folder name
    - and so on
    - files with tag "remove" are skipped
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect PNG files under <input-root> and copy files with the same "
            "tag into <output-root>/<tag>/"
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory to search PNG files from",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory. Files will be copied to <output-root>/<tag>/",
    )
    parser.add_argument(
        "--tag-level",
        type=int,
        default=1,
        help=(
            "Which ancestor folder should be treated as the tag. "
            "1 = parent folder, 2 = grandparent folder"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without copying files",
    )
    return parser.parse_args()


def iter_png_files(input_root: Path) -> list[Path]:
    # Recursively collect .png files under the input root.
    # 遞迴收集 input root 底下所有 .png 檔案。
    return sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() == ".png")


def resolve_tag(file_path: Path, input_root: Path, tag_level: int) -> str:
    # Resolve tag from the file's relative ancestor folders.
    # 從檔案相對於 input root 的祖先資料夾中取出 tag。
    relative_path = file_path.relative_to(input_root)
    parent_parts = relative_path.parts[:-1]
    if len(parent_parts) < tag_level:
        raise ValueError(
            f"File '{file_path}' does not have enough parent levels for --tag-level {tag_level}"
        )

    return parent_parts[-tag_level]


def build_unique_output_path(output_dir: Path, source_path: Path) -> Path:
    # Avoid overwriting when two files under the same tag share the same name.
    # 如果同一個 tag 內有同名檔案，避免直接覆蓋。
    candidate = output_dir / source_path.name
    if not candidate.exists():
        return candidate

    stem = source_path.stem
    suffix = source_path.suffix
    counter = 1
    while True:
        candidate = output_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if args.tag_level < 1:
        raise ValueError("--tag-level must be >= 1")

    png_files = iter_png_files(input_root)
    if not png_files:
        print(f"No PNG files found under: {input_root}")
        return

    copied_count = 0
    skipped_remove_count = 0
    for png_file in png_files:
        tag = resolve_tag(png_file, input_root, args.tag_level)
        if tag.lower() == "remove":
            skipped_remove_count += 1
            continue

        target_dir = output_root / tag
        target_path = build_unique_output_path(target_dir, png_file)

        print(f"{png_file} -> {target_path}")
        if args.dry_run:
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png_file, target_path)
        copied_count += 1

    if args.dry_run:
        print(f"Dry run complete. Planned {len(png_files)} file(s).")
    else:
        print(f"Copied {copied_count} PNG file(s) to: {output_root}")
    if skipped_remove_count:
        print(f"Skipped {skipped_remove_count} PNG file(s) because tag='remove'.")


if __name__ == "__main__":
    main()
