"""
Check image resolutions in a folder and count how many meet a target resolution.

This script recursively scans all images under the given directory and
categorizes them into:
- High resolution (>= target width x target height, including rotated images)
- Low resolution (< target width x target height)
- Unreadable images (corrupted or unsupported)

Supported image formats:
    .jpg, .jpeg, .png, .bmp, .tif, .tiff

Example:
    python -m scripts.tools.check_image_resolution \
      --input-dir data/images \
      --first 1920 \
      --second 1080

Output:
    >=1920x1080: 120
    <1920x1080: 45
    Unreadable: 3
    =====================
    Result saved in:
      data/images/hi.txt
      data/images/lo.txt
      data/images/bad.txt

Notes:
    - Images are checked in both orientations (width x height and height x width),
      so rotated images (e.g., 1080x1920) are also counted as high resolution.
    - Useful for dataset quality control before training, especially for
      small object detection tasks that require sufficient image resolution.
"""

import argparse
from pathlib import Path
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check image resolutions in a folder"
    )
    parser.add_argument("--input-dir", required=True, help="Image input folder path")
    parser.add_argument("--first", type=int, required=True, help="Target width",default="1920")
    parser.add_argument("--second", type=int, required=True, help="Target height",default="1080")
    args = parser.parse_args()

    root = Path(args.input_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    hi = 0
    lo = 0
    bad = []

    hi_list = ["path\tw\th\n"]
    lo_list = ["path\tw\th\n"]

    for path in root.rglob("*"):
        if path.suffix.lower() not in exts:
            continue

        try:
            with Image.open(path) as img:
                w, h = img.size

            if (w >= args.first and h >= args.second) or (
                h >= args.first and w >= args.second
            ):
                hi += 1
                hi_list.append(f"{path}\t{w}\t{h}\n")
            else:
                lo += 1
                lo_list.append(f"{path}\t{w}\t{h}\n")

        except Exception as e:
            bad.append(f"{path}\t{e}\n")

    print(f">={args.first}x{args.second}: {hi}")
    print(f"<{args.first}x{args.second}: {lo}")
    print(f"Unreadable: {len(bad)}")

    hi_path = root / "hi.txt"
    lo_path = root / "lo.txt"
    bad_path = root / "bad.txt"

    with open(hi_path, "w", encoding="utf-8") as f:
        f.writelines(hi_list)

    with open(lo_path, "w", encoding="utf-8") as f:
        f.writelines(lo_list)

    with open(bad_path, "w", encoding="utf-8") as f:
        f.write("path\terror\n")
        f.writelines(bad)

    print("=====================")
    print(f"Result saved in {hi_path} {lo_path} {bad_path}")


if __name__ == "__main__":
    main()