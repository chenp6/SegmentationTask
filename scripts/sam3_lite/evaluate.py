from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from .dataset import load_coco, normalize_prompt_text, resolve_image_path, resolve_split


DEFAULT_DATA_ROOT = "data/ward_dataset_split/content/dataset_split"


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def scale_clip_xyxy(box_xyxy: np.ndarray, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    return [x1, y1, x2, y2]


def run_prediction(
    *,
    data_root: Path,
    split: str,
    checkpoint: str,
    output_json: Path,
    threshold: float,
    mask_threshold: float,
    max_images: int | None,
    device: str | None,
) -> Path:
    split_name = resolve_split(data_root, split)
    split_dir = data_root / split_name
    coco = load_coco(split_dir / "_annotations.coco.json")
    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
    images = coco.get("images", [])
    if max_images is not None and max_images > 0:
        images = images[:max_images]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    model = AutoModel.from_pretrained(checkpoint)
    processor = AutoProcessor.from_pretrained(checkpoint)
    model.to(torch_device)
    model.eval()

    results: list[dict] = []
    with torch.no_grad():
        for img_info in tqdm(images, desc=f"Predict [{split_name}]"):
            image_id = int(img_info["id"])
            image_path = resolve_image_path(split_dir, str(img_info["file_name"]))
            image = Image.open(image_path).convert("RGB")
            h, w = image.height, image.width

            for cat in categories:
                cat_id = int(cat["id"])
                text_prompt = normalize_prompt_text(str(cat.get("name", cat_id)))

                inputs = processor(images=image, text=text_prompt, return_tensors="pt")
                inputs = {k: v.to(torch_device) for k, v in inputs.items()}
                outputs = model(**inputs)

                pred_masks = outputs.pred_masks
                pred_boxes = outputs.pred_boxes
                pred_logits = outputs.pred_logits
                presence_logits = outputs.presence_logits
                if pred_masks is None or pred_boxes is None:
                    continue

                masks_prob = pred_masks.sigmoid()
                masks_up = F.interpolate(masks_prob, size=(h, w), mode="bilinear", align_corners=False)
                masks_bin = (masks_up > mask_threshold)[0].detach().cpu().numpy().astype(np.uint8)

                if pred_logits is not None:
                    scores = pred_logits.sigmoid()
                else:
                    scores = torch.ones((1, pred_masks.shape[1]), device=pred_masks.device)
                if presence_logits is not None:
                    scores = scores * presence_logits.sigmoid()

                scores_np = scores[0].detach().float().cpu().numpy()
                boxes_np = pred_boxes[0].detach().float().cpu().numpy()

                for score, box_xyxy, mask_bin in zip(scores_np, boxes_np, masks_bin):
                    if float(score) <= threshold:
                        continue
                    if mask_bin.sum() == 0:
                        continue

                    x1, y1, x2, y2 = scale_clip_xyxy(box_xyxy, w, h)
                    bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        continue

                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": cat_id,
                            "bbox": bbox,
                            "segmentation": mask_to_rle(mask_bin),
                            "score": float(score),
                        }
                    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"Saved predictions: {output_json}")
    print(f"Total predictions: {len(results)}")
    return output_json


def evaluate_coco(ann_path: Path, pred_json: Path, iou_type: str) -> list[float]:
    coco_gt = COCO(str(ann_path))
    coco_dt = coco_gt.loadRes(str(pred_json))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return list(coco_eval.stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM3-LiteText on COCO")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, default="output/sam3_lite/best")
    parser.add_argument("--pred-json", type=str, default=None, help="Use existing predictions if provided")
    parser.add_argument("--output-json", type=str, default="output/sam3_lite/pred_test.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_name = resolve_split(data_root, args.split)
    ann_path = data_root / split_name / "_annotations.coco.json"

    if args.pred_json:
        pred_json = Path(args.pred_json)
    else:
        pred_json = run_prediction(
            data_root=data_root,
            split=split_name,
            checkpoint=args.checkpoint,
            output_json=Path(args.output_json),
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            max_images=args.max_images,
            device=args.device,
        )

    print("\n[COCO Eval] bbox")
    bbox_stats = evaluate_coco(ann_path, pred_json, "bbox")
    print("\n[COCO Eval] segm")
    segm_stats = evaluate_coco(ann_path, pred_json, "segm")

    print("\nSummary")
    print(f"  pred_json: {pred_json}")
    print(f"  bbox mAP50-95: {bbox_stats[0]:.4f}")
    print(f"  bbox mAP50:    {bbox_stats[1]:.4f}")
    print(f"  segm mAP50-95: {segm_stats[0]:.4f}")
    print(f"  segm mAP50:    {segm_stats[1]:.4f}")


if __name__ == "__main__":
    main()
