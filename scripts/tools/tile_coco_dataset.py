"""
Tile a COCO detection or instance-segmentation dataset into smaller crops.

This tool supports:
- COCO detection annotations with bbox only
- COCO instance-segmentation annotations with polygon or RLE segmentation

Input layout:
    input_root/
      train/_annotations.coco.json
      valid/_annotations.coco.json
      test/_annotations.coco.json

Output layout:
    output_root/
      train/_annotations.coco.json
      train/*.jpg|png
      valid/_annotations.coco.json
      valid/*.jpg|png
      test/_annotations.coco.json
      test/*.jpg|png

Example:
    python -m scripts.tools.tile_coco_dataset \
      --input-root data/hiod_coco \
      --output-root data/hiod_coco_tiled \
      --tile-width 896 \
      --tile-height 896 \
      --stride-x 640 \
      --stride-y 640 \
      --splits train \
      --copy-unprocessed-splits
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_json(path: Path) -> dict:
    # 讀取單一 JSON 檔。
    # Read one JSON file from disk.
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    # 將結果 JSON 寫回硬碟，並自動建立父資料夾。
    # Save the output JSON and create parent directories if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def anns_by_image_id(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    # 依 image_id 將 annotations 分組，方便後續逐張圖處理。
    # Group annotations by image_id so each image can be tiled independently.
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        grouped.setdefault(int(ann["image_id"]), []).append(ann)
    return grouped


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    # Roboflow 匯出的 file_name 有時會帶相對路徑，有時只有檔名，
    # 這裡先嘗試原樣路徑，再退回只比對檔名。
    # Roboflow exports may store either a relative path or just the basename,
    # so try both before failing.
    direct = split_dir / file_name
    if direct.exists():
        return direct

    by_name = split_dir / Path(file_name).name
    if by_name.exists():
        return by_name

    raise FileNotFoundError(f"Image not found under {split_dir}: {file_name}")


def segmentation_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    # 將 COCO polygon / RLE segmentation 統一轉成二值 mask，
    # 方便後面在 tile 區域內裁切並重新計算標註。
    # Convert COCO polygon / RLE segmentation into one binary mask so it can
    # be cropped to the tile window and rebuilt afterward.
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


def binary_mask_to_polygons(mask: np.ndarray, min_area: float = 1.0) -> List[List[float]]:
    # 把 tile 內的 binary mask 重新轉回 COCO polygon。
    # Convert the tile-local binary mask back into COCO polygon format.
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygon = points.astype(np.float32).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def binary_mask_to_rle(mask: np.ndarray) -> dict:
    # 當使用者指定輸出 RLE 時，將 tile mask 編碼成 COCO 相容格式。
    # Encode the tile mask as COCO-compatible RLE when requested.
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def mask_to_bbox_xywh(mask: np.ndarray) -> List[float] | None:
    # 從裁切後的 mask 重新計算最小外接框，維持 COCO 的 xywh 格式。
    # Recompute the tight bounding box from the cropped mask in COCO xywh format.
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def clip_bbox_to_tile(bbox: List[float], tile_x: int, tile_y: int, tile_w: int, tile_h: int) -> List[float] | None:
    # 將原始 bbox 裁切到 tile 範圍內，並改寫成 tile-local 座標。
    # Clip the original bbox to the tile window and shift it into tile-local coordinates.
    x, y, w, h = [float(v) for v in bbox]
    x1 = max(x, tile_x)
    y1 = max(y, tile_y)
    x2 = min(x + w, tile_x + tile_w)
    y2 = min(y + h, tile_y + tile_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 - tile_x, y1 - tile_y, x2 - x1, y2 - y1]


def compute_tile_starts(full_size: int, tile_size: int, stride: int) -> List[int]:
    # 計算每個 tile 的起始座標，並確保最後一塊會貼齊影像邊界，
    # 避免尾端區域因 stride 不能整除而被漏掉。
    # Compute tile start positions and force the final tile to touch the image edge
    # so the trailing region is still covered even when stride does not divide evenly.
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")

    if tile_size >= full_size:
        return [0]

    starts = list(range(0, max(1, full_size - tile_size + 1), stride))
    last_start = full_size - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def tile_image(image: Image.Image, tile_x: int, tile_y: int, tile_w: int, tile_h: int) -> Image.Image:
    # 依指定視窗從原圖裁出一個 tile。
    # Crop one tile window from the source image.
    return image.crop((tile_x, tile_y, tile_x + tile_w, tile_y + tile_h))


def copy_split_without_tiling(split: str, input_root: Path, output_root: Path) -> None:
    # 將未指定 tile 的 split 原樣複製到輸出資料夾，包含 annotation 與影像。
    # Copy an untouched split into the output root, including annotations and images.
    src_split_dir = input_root / split
    ann_path = src_split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return

    dst_split_dir = output_root / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ann_path, dst_split_dir / "_annotations.coco.json")

    copied_images = 0
    for image_path in src_split_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        shutil.copy2(image_path, dst_split_dir / image_path.name)
        copied_images += 1

    print(f"\n[{split}] copied without tiling")
    print(f"  images copied: {copied_images}")
    print(f"  output:        {dst_split_dir}")


def build_tiled_annotation(
    ann: dict,
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    image_height: int,
    image_width: int,
    min_bbox_area: float,
    min_mask_area: int,
    segmentation_format: str,
) -> dict | None:
    seg = ann.get("segmentation")

    if seg:
        # segmentation dataset: 先把整張圖的 instance mask 解開，
        # 再裁成 tile 內的 mask，最後重建 tile 專屬的 bbox / segmentation。
        # Segmentation case: decode the full-image instance mask, crop it to the tile,
        # then rebuild tile-specific bbox / segmentation fields.
        full_mask = segmentation_to_binary_mask(seg, height=image_height, width=image_width)
        tile_mask = full_mask[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
        mask_area = int(tile_mask.sum())
        if mask_area < min_mask_area:
            return None

        bbox_xywh = mask_to_bbox_xywh(tile_mask)
        if bbox_xywh is None or bbox_xywh[2] * bbox_xywh[3] < min_bbox_area:
            return None

        if segmentation_format == "rle":
            tiled_segmentation = binary_mask_to_rle(tile_mask)
        else:
            polygons = binary_mask_to_polygons(tile_mask, min_area=1.0)
            if not polygons:
                return None
            tiled_segmentation = polygons

        new_ann = dict(ann)
        new_ann["bbox"] = bbox_xywh
        new_ann["area"] = float(mask_area)
        new_ann["segmentation"] = tiled_segmentation
        new_ann["iscrowd"] = 0
        return new_ann

    bbox_xywh = ann.get("bbox")
    if not bbox_xywh:
        return None

    # detection dataset: 沒有 segmentation 時，只需要把 bbox 裁切到 tile 內。
    # Detection case: if no segmentation exists, only clip the bbox into the tile.
    clipped_bbox = clip_bbox_to_tile(bbox_xywh, tile_x=tile_x, tile_y=tile_y, tile_w=tile_w, tile_h=tile_h)
    if clipped_bbox is None or clipped_bbox[2] * clipped_bbox[3] < min_bbox_area:
        return None

    new_ann = dict(ann)
    new_ann["bbox"] = clipped_bbox
    new_ann["area"] = float(clipped_bbox[2] * clipped_bbox[3])
    return new_ann


def process_split(
    split: str,
    input_root: Path,
    output_root: Path,
    tile_width: int,
    tile_height: int,
    stride_x: int,
    stride_y: int,
    min_bbox_area: float,
    min_mask_area: int,
    keep_empty_tiles: bool,
    segmentation_format: str,
    image_format: str,
) -> None:
    # 逐個 split 處理：讀原始 COCO、切 tile、重建 images / annotations，
    # 最後輸出成新的 tiled COCO dataset。
    # Process one split end-to-end: read COCO, tile images, rebuild images / annotations,
    # and write a new tiled COCO dataset.
    split_dir = input_root / split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"Skipping split '{split}': annotation file not found at {ann_path}")
        return

    coco = load_json(ann_path)
    images = coco.get("images", [])
    categories = coco.get("categories", [])
    grouped_annotations = anns_by_image_id(coco.get("annotations", []))

    out_split_dir = output_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    next_image_id = 1
    next_ann_id = 1
    saved_tile_count = 0

    for image_info in images:
        # 對每張原圖建立一組滑動視窗；stride 決定相鄰 tile 的重疊程度。
        # Build a sliding window over each source image; stride controls overlap between tiles.
        image_id = int(image_info["id"])
        image_path = resolve_image_path(split_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        xs = compute_tile_starts(image_width, tile_width, stride_x)
        ys = compute_tile_starts(image_height, tile_height, stride_y)

        for tile_y in ys:
            for tile_x in xs:
                tile = tile_image(image, tile_x, tile_y, tile_width, tile_height)
                tiled_anns: list[dict] = []

                for ann in grouped_annotations.get(image_id, []):
                    # 只保留和目前 tile 有交集、且裁切後仍然有效的 annotation。
                    # Keep only annotations that intersect this tile and remain valid after cropping.
                    new_ann = build_tiled_annotation(
                        ann=ann,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        tile_w=tile_width,
                        tile_h=tile_height,
                        image_height=image_height,
                        image_width=image_width,
                        min_bbox_area=min_bbox_area,
                        min_mask_area=min_mask_area,
                        segmentation_format=segmentation_format,
                    )
                    if new_ann is None:
                        continue
                    new_ann["id"] = next_ann_id
                    next_ann_id += 1
                    tiled_anns.append(new_ann)

                if not tiled_anns and not keep_empty_tiles:
                    continue

                # 輸出 tile 影像，並同步建立新的 image metadata。
                # Save the tile image and create its corresponding image metadata entry.
                source_stem = Path(image_info["file_name"]).stem
                tile_file_name = f"{source_stem}__x{tile_x}_y{tile_y}_w{tile_width}_h{tile_height}.{image_format}"
                tile_path = out_split_dir / tile_file_name
                tile.save(tile_path)

                new_image_info = dict(image_info)
                new_image_info["id"] = next_image_id
                new_image_info["file_name"] = tile_file_name
                new_image_info["width"] = tile_width
                new_image_info["height"] = tile_height
                new_images.append(new_image_info)

                for ann in tiled_anns:
                    ann["image_id"] = next_image_id
                    new_annotations.append(ann)

                next_image_id += 1
                saved_tile_count += 1

    new_coco = dict(coco)
    new_coco["images"] = new_images
    new_coco["annotations"] = new_annotations
    new_coco["categories"] = categories

    out_ann_path = out_split_dir / "_annotations.coco.json"
    save_json(out_ann_path, new_coco)

    print(f"\n[{split}] done")
    print(f"  tiles saved:        {saved_tile_count}")
    print(f"  images in json:     {len(new_images)}")
    print(f"  annotations in json:{len(new_annotations)}")
    print(f"  output:             {out_ann_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile a COCO detection or segmentation dataset")
    parser.add_argument("--input-root", required=True, help="Input COCO dataset root")
    parser.add_argument("--output-root", required=True, help="Output tiled dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits to process")
    parser.add_argument("--tile-width", type=int, required=True, help="Tile width in pixels")
    parser.add_argument("--tile-height", type=int, required=True, help="Tile height in pixels")
    parser.add_argument("--stride-x", type=int, default=None, help="Horizontal stride; defaults to tile width")
    parser.add_argument("--stride-y", type=int, default=None, help="Vertical stride; defaults to tile height")
    parser.add_argument("--min-bbox-area", type=float, default=4.0, help="Drop tiled annotations smaller than this bbox area")
    parser.add_argument("--min-mask-area", type=int, default=4, help="Drop tiled segmentation masks smaller than this pixel area")
    parser.add_argument("--keep-empty-tiles", action="store_true", help="Keep tiles with no annotations")
    parser.add_argument("--copy-unprocessed-splits", action="store_true", help="Copy splits not listed in --splits without tiling")
    parser.add_argument("--segmentation-format", choices=["polygon", "rle"], default="polygon", help="Output format for segmentation annotations")
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg", help="Image format for tiled crops")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, stride_x: int, stride_y: int) -> None:
    # 基本參數檢查，避免產生無效 tile 設定。
    # Validate basic numeric arguments before doing any heavy processing.
    if args.tile_width <= 0 or args.tile_height <= 0:
        raise ValueError("tile width and tile height must both be > 0")
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("stride_x and stride_y must both be > 0")
    if args.min_bbox_area < 0:
        raise ValueError("min_bbox_area must be >= 0")
    if args.min_mask_area < 0:
        raise ValueError("min_mask_area must be >= 0")


def main() -> None:
    args = parse_args()

    # 若未指定 stride，就預設為不重疊切圖：stride == tile size。
    # If stride is omitted, default to non-overlapping tiling: stride == tile size.
    stride_x = args.stride_x if args.stride_x is not None else args.tile_width
    stride_y = args.stride_y if args.stride_y is not None else args.tile_height
    validate_args(args, stride_x=stride_x, stride_y=stride_y)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        process_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            stride_x=stride_x,
            stride_y=stride_y,
            min_bbox_area=args.min_bbox_area,
            min_mask_area=args.min_mask_area,
            keep_empty_tiles=args.keep_empty_tiles,
            segmentation_format=args.segmentation_format,
            image_format=args.image_format,
        )

    if args.copy_unprocessed_splits:
        # 若只想 tile 部分 split（例如 train），這裡會把其餘 split
        # 直接原樣複製到 output_root，讓整個資料集結構保持完整。
        # If only some splits are tiled (for example train only), copy the
        # remaining splits through unchanged so the output dataset stays complete.
        requested_splits = set(args.splits)
        for split in ["train", "valid", "test"]:
            if split in requested_splits:
                continue
            copy_split_without_tiling(split=split, input_root=input_root, output_root=output_root)


if __name__ == "__main__":
    main()
