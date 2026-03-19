"""
Extract per-instance transparent PNGs from a COCO instance segmentation dataset.

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
    python -m scripts.object_detection_to_image_segmentaion.extract_coco_instances_to_png \
    --input-root data/hiod_coco \
    --output-root output/hiod_instances_png \
    --splits train
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
    # 讀取 COCO annotation JSON。
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def anns_by_image_id(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    # 把 annotations 依 image_id 分組，後續處理單張影像時比較快查。
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    # 優先使用 annotation 內的原始 file_name，找不到時再退回只比對檔名或副檔名。
    direct = split_dir / file_name
    if direct.exists():
        return direct

    by_name = split_dir / Path(file_name).name
    if by_name.exists():
        return by_name

    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found under {split_dir}: {file_name}")


def segmentation_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    # COCO segmentation 可能是 polygon list 或 RLE dict，這裡統一轉成 (H, W) binary mask。
    if isinstance(segmentation, list): #polygon list
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict): #RLE dict
        rle = segmentation
    else:
        return np.zeros((height, width), dtype=np.uint8)

    decoded = coco_mask.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def mask_to_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    # 從 binary mask 取出最小外接框，供後續裁切 instance 使用。
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_xyxy(box: tuple[int, int, int, int], width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    # 在 instance 外接框周圍補一些 padding，避免裁切太緊。
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def slugify(value: str) -> str:
    # 把類別名或檔名整理成安全的輸出路徑字串。
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("._") or "unknown"


def build_rgba_crop(image_rgb: np.ndarray, mask: np.ndarray, crop_box: tuple[int, int, int, int]) -> Image.Image:
    # 將裁切區域組成 RGBA 影像，alpha 直接使用 instance mask，達成去背效果。
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
    # 逐個 split 處理 COCO segmentation dataset，輸出每個 instance 的透明 PNG。
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
        # 先讀原圖，這樣同一張圖底下的多個 instance 可以重複使用同一份像素資料。
        image_path = resolve_image_path(split_dir, image_info["file_name"])
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        height, width = image_rgb.shape[:2]

        for ann in grouped_annotations.get(image_id, []):
            # crowd annotation 不適合做單一 instance 匯出，直接略過。
            if ann.get("iscrowd", 0):
                skipped_count += 1
                continue

            mask = segmentation_to_binary_mask(ann.get("segmentation"), height=height, width=width)
            mask_area = int(mask.sum())
            # 太小的 mask 通常噪聲較多，直接略過避免輸出品質太差。
            if mask_area < min_mask_area:
                skipped_count += 1
                continue

            box = mask_to_xyxy(mask)
            if box is None:
                skipped_count += 1
                continue

            crop_box = expand_xyxy(box, width=width, height=height, padding=padding)
            rgba_image = build_rgba_crop(image_rgb=image_rgb, mask=mask, crop_box=crop_box)

            category = categories_by_id.get(int(ann["category_id"]), {})
            category_name = slugify(str(category.get("name", ann["category_id"])))
            image_stem = slugify(Path(image_info["file_name"]).stem)
            out_dir = output_root / split / category_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # 檔名保留原圖 stem、annotation id、category id，方便回查來源。
            file_stem = f"{image_stem}__ann{ann['id']}__cat{ann['category_id']}"
            png_path = out_dir / f"{file_stem}.png"
            rgba_image.save(png_path)

            mask_path = None
            if save_full_mask:
                # 視需要另外輸出完整尺寸的 binary mask，方便除錯或後續流程使用。
                mask_dir = output_root / split / "_masks"
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_path = mask_dir / f"{file_stem}.png"
                Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_path)

            # manifest 記錄輸出 PNG 與原始 annotation 的對應資訊。
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
    # CLI 參數集中在這裡，方便後續獨立當工具腳本使用。
    parser = argparse.ArgumentParser(description="Extract per-instance transparent PNGs from a COCO segmentation dataset")
    parser.add_argument("--input-root", required=True, help="COCO segmentation dataset root")
    parser.add_argument("--output-root", default="output/instance_pngs", help="Directory to save instance PNGs")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--padding", type=int, default=0, help="Extra pixel padding around each instance crop")
    parser.add_argument("--min-mask-area", type=int, default=0, help="Skip instances smaller than this mask area")
    parser.add_argument("--save-full-mask", action="store_true", help="Also save one full-size binary mask PNG per instance")
    return parser.parse_args()


def main() -> None:
    # 主流程：逐個 split 輸出 instance PNG，最後產出一份 manifest.jsonl。
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
