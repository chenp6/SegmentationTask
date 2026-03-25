"""
Convert a COCO object detection dataset into a COCO instance segmentation dataset
using a pretrained SAM2 model prompted by existing bounding boxes.

Input layout:
    input_root/
      train/_annotations.coco.json
      valid/_annotations.coco.json
      test/_annotations.coco.json

Output layout:
    output_root/
      train/_annotations.coco.json
      valid/_annotations.coco.json
      test/_annotations.coco.json

Images are hard-linked when possible and copied as a fallback.

Example:
    python -m scripts.object_detection_to_image_segmentaion.convert_coco_detection_to_segmentation \
      --input-root data/hiod_coco \
      --output-root data/hiod_sam2_seg \
      --sam2-model-id facebook/sam2.1-hiera-large \
      --device cuda


workflow
讀取 COCO detection json
    ↓
逐張圖片處理
    ↓
對每個 annotation 取出 bbox
    ↓
bbox 轉成 [x1, y1, x2, y2]
    ↓
可選擇稍微放大 bbox
    ↓
把 bbox 當 SAM2 prompt
    ↓
SAM2 回傳 mask
    ↓
mask 轉成 polygon 或 RLE
    ↓
寫回新的 COCO annotation
    ↓
存成新的 segmentation json
"""
import argparse
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_coco(ann_path: Path) -> dict:
    # 讀取 COCO 格式標註 JSON。
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(ann_path: Path, data: dict) -> None:
    # 將轉換後的 COCO segmentation 標註寫回硬碟。
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def anns_by_image_id(annotations: Iterable[dict]) -> Dict[int, List[dict]]:
    # 依 image_id 分組 annotation，方便後面逐張圖取出對應物件。
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def _link_or_copy_image(src: Path, dst: Path, force_copy: bool = False) -> None:
    # 優先用 hard link 保留空間；若檔案系統不支援，再退回 copy。
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def bbox_xywh_to_xyxy(bbox: List[float], width: int, height: int) -> np.ndarray:
    # 將 COCO bbox 格式 [x, y, w, h] 轉成 [x1, y1, x2, y2]，並裁切到影像邊界內。
    x, y, w, h = bbox
    x1 = max(0.0, min(float(x), width - 1))
    y1 = max(0.0, min(float(y), height - 1))
    x2 = max(x1 + 1.0, min(float(x + w), width))
    y2 = max(y1 + 1.0, min(float(y + h), height))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_box_xyxy(box_xyxy: np.ndarray, width: int, height: int, pad_ratio: float) -> np.ndarray:
    # 視需要把 bbox 再往外擴一點，讓 SAM2 prompt 能包含更多物件邊界資訊。
    if pad_ratio <= 0:
        return box_xyxy.astype(np.float32)
    x1, y1, x2, y2 = box_xyxy.tolist()
    box_w = x2 - x1
    box_h = y2 - y1
    x_pad = box_w * pad_ratio
    y_pad = box_h * pad_ratio
    return np.array(
        [
            max(0.0, x1 - x_pad),
            max(0.0, y1 - y_pad),
            min(float(width), x2 + x_pad),
            min(float(height), y2 + y_pad),
        ],
        dtype=np.float32,
    )


def mask_to_xywh(mask: np.ndarray) -> List[float]:
    # 從二值 mask 重新計算最小外接 bbox，輸出仍維持 COCO 的 [x, y, w, h] 格式。
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def box_xyxy_to_polygon(box_xyxy: np.ndarray) -> List[float]:
    # 當 SAM2 沒有產出有效 mask 時，用矩形框生成一個最基本的 polygon fallback。
    x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def binary_mask_to_polygons(mask: np.ndarray, min_area: float = 16.0) -> List[List[float]]:
    # 把二值 mask 轉成 COCO polygon segmentation；太小的區塊會被忽略。
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
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
    # 需要 RLE 輸出時，將 mask 編成 COCO 相容的 RLE 格式。
    from pycocotools import mask as coco_mask

    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


class SAM2BoxSegmenter:
    def __init__(
        self,
        model_id: str,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        device: str | None = None,
        multimask_output: bool = True,
    ):
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.multimask_output = multimask_output
        # 建立 SAM2 predictor，後續會重複拿來做 bbox prompt segmentation。
        self.predictor = self._build_predictor()

    def _build_predictor(self):
        # 優先支援本地 checkpoint + config，其次嘗試 Hugging Face 預訓練模型。
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if self.checkpoint_path and self.config_path:
            from sam2.build_sam import build_sam2

            sam2_model = build_sam2(
                self.config_path,
                self.checkpoint_path,
                device=self.device,
            )
            return SAM2ImagePredictor(sam2_model)

        try:
            return SAM2ImagePredictor.from_pretrained(self.model_id, device=self.device)
        except Exception:
            from sam2.build_sam import build_sam2_hf

            sam2_model = build_sam2_hf(self.model_id, device=self.device)
            return SAM2ImagePredictor(sam2_model)

    def set_image(self, image_rgb: np.ndarray) -> None:
        # 每處理一張圖都先把影像送進 predictor，之後同張圖可重複用不同 bbox 查詢。
        self.predictor.set_image(image_rgb)

    def predict_mask(self, box_xyxy: np.ndarray) -> np.ndarray:
        # 用 bbox 當 prompt 做單物件 segmentation，必要時在 GPU 上啟用 autocast。
        with torch.inference_mode():
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if str(self.device).startswith("cuda")
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                try:
                    masks, scores, _ = self.predictor.predict(
                        box=box_xyxy,
                        multimask_output=self.multimask_output,
                    )
                except Exception:
                    masks, scores, _ = self.predictor.predict(
                        box=box_xyxy[None, :],
                        multimask_output=self.multimask_output,
                    )

        masks = np.asarray(masks)
        scores = np.asarray(scores) if scores is not None else None

        if masks.ndim == 2:
            return masks.astype(np.uint8)

        if masks.ndim == 3 and len(masks) > 0:
            # multimask 模式下會回傳多個候選 mask，這裡挑分數最高的那一個。
            best_idx = int(np.argmax(scores)) if scores is not None and len(scores) == len(masks) else 0
            return masks[best_idx].astype(np.uint8)

        raise RuntimeError(f"Unexpected SAM2 mask shape: {masks.shape}")


def build_segmentation(
    mask: np.ndarray,
    segmentation_format: str,
    fallback_box: np.ndarray,
) -> tuple[list | dict, float, List[float]]:
    # 依需求輸出 RLE 或 polygon；若 polygon 轉不出來，就退回矩形 polygon。
    if segmentation_format == "rle":
        return binary_mask_to_rle(mask), float(mask.sum()), mask_to_xywh(mask)

    polygons = binary_mask_to_polygons(mask)
    if polygons:
        return polygons, float(mask.sum()), mask_to_xywh(mask)

    rectangle = [box_xyxy_to_polygon(fallback_box)]
    area = float(max(1.0, (fallback_box[2] - fallback_box[0]) * (fallback_box[3] - fallback_box[1])))
    bbox = [
        float(fallback_box[0]),
        float(fallback_box[1]),
        float(fallback_box[2] - fallback_box[0]),
        float(fallback_box[3] - fallback_box[1]),
    ]
    return rectangle, area, bbox


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    # Roboflow 匯出的 file_name 可能帶路徑也可能只有檔名，這裡做幾種常見查找。
    direct = split_dir / file_name
    if direct.exists():
        return direct

    candidate = split_dir / Path(file_name).name
    if candidate.exists():
        return candidate

    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        image_path = split_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path

    raise FileNotFoundError(f"Image not found under {split_dir}: {file_name}")


def convert_split(
    split: str,
    input_root: Path,
    output_root: Path,
    segmenter: SAM2BoxSegmenter,
    segmentation_format: str,
    box_pad_ratio: float,
    empty_mask_policy: str,
    min_mask_area: int,
    max_images: int | None,
    force_copy_images: bool,
) -> None:
    # 轉換單一 split，例如 train / valid / test。
    split_dir = input_root / split
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        print(f"Skipping split '{split}': annotation file not found at {ann_path}")
        return

    coco = load_coco(ann_path)
    images = coco.get("images", [])
    if max_images is not None:
        # 方便先做小批量 smoke test。
        images = images[:max_images]

    anns_grouped = anns_by_image_id(coco.get("annotations", []))
    processed_images = []
    new_annotations = []

    total_anns = 0
    fallback_count = 0
    skipped_count = 0

    for image_info in tqdm(images, desc=f"Converting [{split}]"):
        # 讀取原始影像，並先讓 SAM2 predictor 建立這張圖的內部表徵。
        image_path = resolve_image_path(split_dir, image_info["file_name"])
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        height, width = image_rgb.shape[:2]
        segmenter.set_image(image_rgb)

        # 輸出資料集沿用原圖；預設 hard-link，必要時才 copy。
        out_split_dir = output_root / split
        out_image_path = out_split_dir / Path(image_info["file_name"]).name
        _link_or_copy_image(image_path, out_image_path, force_copy=force_copy_images)

        # 更新 image metadata，讓新的 COCO 標註指向輸出資料夾中的影像檔名。
        new_image_info = dict(image_info)
        new_image_info["file_name"] = out_image_path.name
        processed_images.append(new_image_info)

        for ann in anns_grouped.get(image_info["id"], []):
            # crowd 標註通常不是單一清楚 instance，不適合直接拿 bbox 做 SAM2 分割。
            if ann.get("iscrowd", 0):
                continue

            total_anns += 1
            # Convert COCO bbox [x, y, w, h] into SAM2-friendly [x1, y1, x2, y2],
            # then optionally expand it a bit so the prompt includes object boundaries.
            original_box = bbox_xywh_to_xyxy(ann["bbox"], width=width, height=height)
            prompt_box = expand_box_xyxy(original_box, width=width, height=height, pad_ratio=box_pad_ratio)

            mask = None
            try:
                # 用 bbox prompt 呼叫 SAM2，取得這個 annotation 對應的 mask。
                mask = segmenter.predict_mask(prompt_box)
                mask = (mask > 0).astype(np.uint8)
            except Exception as exc:
                print(f"Warning: SAM2 failed on image {image_info['file_name']} ann {ann['id']}: {exc}")

            if mask is None or int(mask.sum()) < min_mask_area:
                # 若 SAM2 沒有產出有效遮罩，可選擇跳過，或退回成 bbox 形狀的簡單 mask。
                if empty_mask_policy == "skip":
                    skipped_count += 1
                    continue

                fallback_count += 1
                mask = np.zeros((height, width), dtype=np.uint8)
                x1, y1, x2, y2 = [int(v) for v in prompt_box.tolist()]
                mask[y1:y2, x1:x2] = 1

            # 將最終 mask 轉成 COCO segmentation + area + bbox。
            segmentation, area, bbox_xywh = build_segmentation(
                mask=mask,
                segmentation_format=segmentation_format,
                fallback_box=prompt_box,
            )

            # 保留原 annotation 的類別與 id，僅替換成 segmentation 相關欄位。
            new_ann = dict(ann)
            new_ann["segmentation"] = segmentation
            new_ann["area"] = area
            new_ann["bbox"] = bbox_xywh
            new_ann["iscrowd"] = 0
            new_annotations.append(new_ann)

    # 以原始 COCO 結構為基底，替換成新的 images / annotations 後輸出。
    new_coco = dict(coco)
    new_coco["images"] = processed_images
    new_coco["annotations"] = new_annotations

    out_ann_path = output_root / split / "_annotations.coco.json"
    save_coco(out_ann_path, new_coco)

    print(f"\n[{split}] done")
    print(f"  images:      {len(processed_images)}")
    print(f"  annotations: {len(new_annotations)} / {total_anns}")
    print(f"  fallbacks:   {fallback_count}")
    print(f"  skipped:     {skipped_count}")
    print(f"  output:      {out_ann_path}")


def main() -> None:
    # CLI 入口：讀參數、建立 SAM2 predictor，然後逐個 split 轉換。
    parser = argparse.ArgumentParser(
        description="Use SAM2 box prompts to convert COCO detection into COCO segmentation"
    )
    parser.add_argument("--input-root", required=True, help="COCO detection dataset root")
    parser.add_argument("--output-root", required=True, help="Output COCO segmentation root")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--sam2-checkpoint", default=None, help="Optional local SAM2 checkpoint")
    parser.add_argument("--sam2-config", default=None, help="Optional local SAM2 config")
    parser.add_argument("--device", default=None, help="cuda / cpu; default auto")
    parser.add_argument("--box-pad-ratio", type=float, default=0.0, help="Expand bbox before SAM2 prompt")
    parser.add_argument("--segmentation-format", choices=["polygon", "rle"], default="polygon")
    parser.add_argument("--empty-mask-policy", choices=["box", "skip"], default="box")
    parser.add_argument("--min-mask-area", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None, help="Useful for smoke tests")
    parser.add_argument("--copy-images", action="store_true", help="Copy files instead of hard-linking them")
    parser.add_argument("--single-mask", action="store_true", help="Disable SAM2 multimask output")
    args = parser.parse_args()

    try:
        # 明確檢查 sam2 套件是否已安裝，避免跑到一半才失敗。
        import sam2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The sam2 package is required. Install it with:\n"
            "pip install git+https://github.com/facebookresearch/sam2.git"
        ) from exc

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 先建立一次 SAM2 predictor，整個轉換流程共用。
    segmenter = SAM2BoxSegmenter(
        model_id=args.sam2_model_id,
        checkpoint_path=args.sam2_checkpoint,
        config_path=args.sam2_config,
        device=args.device,
        multimask_output=not args.single_mask,
    )

    for split in args.splits:
        # 逐個 split 轉換，輸出為新的 COCO segmentation dataset。
        convert_split(
            split=split,
            input_root=input_root,
            output_root=output_root,
            segmenter=segmenter,
            segmentation_format=args.segmentation_format,
            box_pad_ratio=args.box_pad_ratio,
            empty_mask_policy=args.empty_mask_policy,
            min_mask_area=args.min_mask_area,
            max_images=args.max_images,
            force_copy_images=args.copy_images,
        )


if __name__ == "__main__":
    main()
