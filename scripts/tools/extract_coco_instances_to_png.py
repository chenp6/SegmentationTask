"""
Extract per-instance transparent PNGs from a COCO instance segmentation dataset.

This file is designed to be standalone so it can be shared from scripts/tools
without depending on the original pipeline package layout.

Input layout:
    input_root/
      train/_annotations.coco.json
      valid/_annotations.coco.json
      test/_annotations.coco.json

Output layout:
    output_root/
      train/<category_name>/*.png
      valid/<category_name>/*.png
      test/<category_name>/*.png
      manifest.jsonl

Example:
    python -m scripts.tools.extract_coco_instances_to_png \
      --input-root data/hiod_sam2_seg \
      --output-root output/hiod_instances_png \
      --padding 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_coco(ann_path: Path) -> dict:
    # 讀取單一 COCO annotation json 檔。
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def anns_by_image_id(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    # 依 image_id 將 annotation 分組，後續可快速取得同一張圖的所有 instance。
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    # 優先使用 COCO file_name 完整路徑；若資料夾層級不同，再退回純檔名比對。
    # 這裡刻意不再用 stem 猜檔名，避免拿到錯圖造成 mask 對位錯誤。
    direct = split_dir / file_name
    if direct.exists():
        return direct

    by_name = split_dir / Path(file_name).name
    if by_name.exists():
        return by_name

    raise FileNotFoundError(f"Image not found under {split_dir}: {file_name}")


def segmentation_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    # COCO segmentation 可能是 polygon list 或 RLE dict，統一轉成 (H, W) binary mask。
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


def mask_to_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    # 由 binary mask 計算前景最小外接框，供裁切 instance 使用。
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_xyxy(box: tuple[int, int, int, int], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    # 在外接框周圍額外補 padding，避免裁切時貼得太緊。
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def slugify(value: str) -> str:
    # 將類別名稱或檔名整理成適合當檔案路徑的字串。
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._") or "unknown"


def build_rgba_crop(image_rgb: np.ndarray, mask: np.ndarray, crop_box: tuple[int, int, int, int]) -> Image.Image:
    # 建立 RGBA 影像，alpha 通道直接使用 instance mask，達成去背效果。
    x1, y1, x2, y2 = crop_box
    rgb_crop = image_rgb[y1:y2, x1:x2]
    alpha_crop = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    rgba = np.dstack([rgb_crop, alpha_crop]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def extract_split(
    split: str,
    input_root: Path,
    output_root: Path,
    padding: int,
    min_mask_area: int,
    save_full_mask: bool,
    manifest_rows: list[dict],
) -> None:
    # 處理單一 split，將每個 instance 匯出成透明背景 PNG。
    split_dir = input_root / split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"Skipping split '{split}': annotation file not found at {ann_path}")
        return

    coco = load_coco(ann_path)
    images_by_id = {int(img["id"]): img for img in coco.get("images", [])}
    categories_by_id = {int(cat["id"]): cat for cat in coco.get("categories", [])}
    grouped_annotations = anns_by_image_id(coco.get("annotations", []))

    saved_count = 0
    skipped_count = 0

    for image_id, image_info in tqdm(images_by_id.items(), desc=f"Extracting [{split}]"):
        # 先讀取原圖，讓同一張圖底下的多個 instance 共用同一份像素資料。
        image_path = resolve_image_path(split_dir, image_info["file_name"])
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        actual_height, actual_width = image_rgb.shape[:2]

        expected_height = int(image_info.get("height", actual_height))
        expected_width = int(image_info.get("width", actual_width))
        # 若 annotation 記錄的尺寸與實際圖片尺寸不同，mask 很容易整體偏移，這裡直接跳過。
        if actual_height != expected_height or actual_width != expected_width:
            print(
                "Warning: image size mismatch for "
                f"{image_info['file_name']}. "
                f"annotation=({expected_width}, {expected_height}) "
                f"actual=({actual_width}, {actual_height}). Skipping this image."
            )
            skipped_count += len(grouped_annotations.get(image_id, []))
            continue

        for ann in grouped_annotations.get(image_id, []):
            # crowd 標註通常不是單一乾淨 instance，不適合直接輸出成單張 PNG。
            if ann.get("iscrowd", 0):
                print(
                    "Warning: iscrowd : 1. Skipping this image."
                )                
                skipped_count += 1
                continue

            # 依 annotation 內的 segmentation 還原出該 instance 的 binary mask。
            mask = segmentation_to_binary_mask(
                ann.get("segmentation"),
                height=expected_height,
                width=expected_width,
            )
            mask_area = int(mask.sum())
            # 太小的 mask 可能只是噪聲或幾乎看不見，依設定直接略過。
            if mask_area < min_mask_area:
                print(
                   f"Warning: min_mask_area < {min_mask_area} for {image_info['file_name']}. Skipping this image."
                )          
                skipped_count += 1
                continue

            box = mask_to_xyxy(mask)
            if box is None:
                print(
                   f"Warning: No valid front-scene for {image_info['file_name']}. Skipping this image."
                )                 
                skipped_count += 1
                continue

            crop_box = expand_xyxy(box, width=actual_width, height=actual_height, padding=padding)
            rgba_image = build_rgba_crop(image_rgb=image_rgb, mask=mask, crop_box=crop_box)

            category = categories_by_id.get(int(ann["category_id"]), {})
            category_name = slugify(str(category.get("name", ann["category_id"])))
            image_stem = slugify(Path(image_info["file_name"]).stem)
            out_dir = output_root / split / category_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # 檔名保留原圖 stem、annotation id、category id，方便後續回查來源。
            file_stem = f"{image_stem}__ann{ann['id']}__cat{ann['category_id']}"
            png_path = out_dir / f"{file_stem}.png"
            rgba_image.save(png_path)

            mask_path = None
            if save_full_mask:
                # 視需要一併輸出完整尺寸的 binary mask，方便除錯或後處理。
                mask_dir = output_root / split / "_masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{file_stem}.png"
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)

            # manifest 記錄每張輸出 PNG 與原始 annotation、圖片的對應資訊。
            manifest_rows.append(
                {
                    "split": split,
                    "image_id": int(image_id),
                    "annotation_id": int(ann["id"]),
                    "category_id": int(ann["category_id"]),
                    "category_name": category.get("name", str(ann["category_id"])),
                    "source_image": str(image_path.as_posix()),
                    "output_png": str(png_path.as_posix()),
                    "mask_png": str(mask_path.as_posix()) if mask_path else None,
                    "mask_area": mask_area,
                    "crop_box_xyxy": [int(v) for v in crop_box],
                    "image_file_name": image_info["file_name"],
                }
            )
            saved_count += 1

    print(f"\n[{split}] done")
    print(f"  saved instances: {saved_count}")
    print(f"  skipped:         {skipped_count}")


def parse_args() -> argparse.Namespace:
    # CLI 參數入口，方便將此腳本獨立當工具使用。
    parser = argparse.ArgumentParser(description="Extract per-instance transparent PNGs from a COCO segmentation dataset")
    parser.add_argument("--input-root", required=True, help="COCO segmentation dataset root")
    parser.add_argument("--output-root", default="output/instance_pngs", help="Directory to save instance PNGs")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--padding", type=int, default=0, help="Extra pixel padding around each instance crop")
    parser.add_argument("--min-mask-area", type=int, default=0, help="Skip instances smaller than this mask area")
    parser.add_argument("--save-full-mask", action="store_true", help="Also save one full-size binary mask PNG per instance")
    return parser.parse_args()


def main() -> None:
    # 主流程：逐個 split 匯出 instance PNG，最後再寫出 manifest.jsonl。
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for split in args.splits:
        extract_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            padding=args.padding,
            min_mask_area=args.min_mask_area,
            save_full_mask=args.save_full_mask,
            manifest_rows=manifest_rows,
        )

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nManifest written to: {manifest_path}")
    print(f"Total instances saved: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
