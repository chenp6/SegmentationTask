"""
Convert a FiftyOne-exported COCO dataset into Roboflow-style COCO split folders.

Expected input (FiftyOne COCODetectionDataset export):
  input_root/
    train/
      data/*.jpg|png
      labels.json
    valid/ or validation/
      data/*.jpg|png
      labels.json
    test/ (optional)
      data/*.jpg|png
      labels.json

Output (Roboflow-style):
  output_root/
    train/
      *.jpg|png
      _annotations.coco.json
    valid/
      *.jpg|png
      _annotations.coco.json
    test/ (optional)
      *.jpg|png
      _annotations.coco.json

Example:
  python -m scripts.tools.from_coco_to_roboflow_dataset \
    --input-root data/coco2017 \
    --output-root data/coco2017_roboflow
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _convert_one_split(src_split_dir: Path, dst_split_dir: Path) -> bool:
    labels_path = src_split_dir / "labels.json"
    data_dir = src_split_dir / "data"

    if not labels_path.exists() or not data_dir.exists():
        return False

    coco = _load_json(labels_path)
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path in sorted(data_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(img_path, dst_split_dir / img_path.name)
            copied += 1

    for img in coco.get("images", []):
        img["file_name"] = Path(img["file_name"]).name

    out_ann = dst_split_dir / "_annotations.coco.json"
    _save_json(out_ann, coco)

    print(f"[ok] {src_split_dir.name:10s} -> {dst_split_dir.name:5s} | images={copied} | ann={out_ann}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FiftyOne COCO export to Roboflow-style COCO splits.")
    parser.add_argument("--input-root", required=True, help="Input root exported by FiftyOne")
    parser.add_argument("--output-root", required=True, help="Output root in Roboflow-style split layout")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Support both 'valid' and 'validation' as source names.
    found_train = _convert_one_split(input_root / "train", output_root / "train")
    found_valid = _convert_one_split(input_root / "valid", output_root / "valid")
    if not found_valid:
        found_valid = _convert_one_split(input_root / "validation", output_root / "valid")
    _convert_one_split(input_root / "test", output_root / "test")

    if not found_train:
        raise FileNotFoundError(f"Missing train split under {input_root}. Expected train/data + train/labels.json")
    if not found_valid:
        raise FileNotFoundError(
            f"Missing valid split under {input_root}. Expected valid/data + valid/labels.json "
            "or validation/data + validation/labels.json"
        )

    print(f"\nDone. Converted dataset at: {output_root}")


if __name__ == "__main__":
    main()

