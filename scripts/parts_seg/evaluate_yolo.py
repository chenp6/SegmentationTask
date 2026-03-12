"""
Evaluate SAM2+YOLO part segmentation (COCO mAP50 / mAP50-95 on val set).

Supports --dataset pascal | ade20k234. Uses the same model and decode logic as sam2_yolo_seg.

Usage (from repo root):
  python -m scripts.parts_seg.evaluate_yolo --checkpoint output/parts_seg_yolo/latest_model.pt --dataset ade20k234
  python -m scripts.parts_seg.evaluate_yolo --checkpoint output/parts_seg_yolo/latest_model.pt --dataset pascal --data_root data/pascal_part
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

try:
    from scripts.sam2_yolo_seg.evaluate_yolo import decode_predictions
    from scripts.sam2_yolo_seg.models.sam2_yolo_model import SAM2YOLO
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from scripts.sam2_yolo_seg.evaluate_yolo import decode_predictions
    from scripts.sam2_yolo_seg.models.sam2_yolo_model import SAM2YOLO

from .dataset import PascalPartDataset, collate_fn as _pascal_collate
from .dataset_ade20k234 import ADE20KPart234Dataset, collate_fn as _ade_collate


def mask_to_rle(binary_mask):
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def build_coco_gt_from_dataset(dataset):
    """Build COCO-format GT from a parts dataset (same image_size as model input)."""
    images = []
    categories = [{"id": c, "name": f"part_{c}"} for c in range(dataset.num_classes)]
    annotations = []
    ann_id = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        masks = sample["masks"]
        labels = sample["labels"]
        H, W = masks.shape[1], masks.shape[2]
        images.append({"id": i, "width": W, "height": H})
        for j in range(masks.shape[0]):
            mask = masks[j].numpy()
            if mask.sum() < 1:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": i,
                "category_id": int(labels[j].item()),
                "segmentation": mask_to_rle(mask),
                "area": int(mask.sum()),
            })
            ann_id += 1
    return {"images": images, "categories": categories, "annotations": annotations}


def evaluate(data_root, checkpoint_path, dataset_name="pascal", split="val", score_thresh=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    image_size = ckpt.get("image_size", 512)

    model = SAM2YOLO.build(num_classes=num_classes, image_size=image_size)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    if dataset_name == "ade20k234":
        val_dataset = ADE20KPart234Dataset(
            data_root, split=split, image_size=image_size, augment=False
        )
    else:
        val_dataset = PascalPartDataset(
            data_root, split=split, image_size=image_size, augment=False
        )

    # Build COCO GT from dataset (masks at image_size)
    gt = build_coco_gt_from_dataset(val_dataset)
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt, f)
        gt_path = f.name
    try:
        coco_gt = COCO(gt_path)
    finally:
        os.unlink(gt_path)

    coco_results = []
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc=f"Eval [{split}]"):
            sample = val_dataset[i]
            image = sample["image"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                outputs = model(image)
            instances = decode_predictions(
                outputs, model.strides, image_size, num_classes,
                score_thresh=score_thresh,
            )
            H, W = sample["masks"].shape[1], sample["masks"].shape[2]
            for inst in instances:
                mask_t = inst["mask"].unsqueeze(0).unsqueeze(0).float()
                mask_resized = F.interpolate(mask_t, (H, W), mode="bilinear", align_corners=False)
                mask_np = (mask_resized.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                coco_results.append({
                    "image_id": i,
                    "category_id": inst["class_idx"],
                    "segmentation": mask_to_rle(mask_np),
                    "score": inst["score"],
                })

    if not coco_results:
        print("No predictions. Try lowering --score_thresh (e.g. 0.1).")
        return
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(f"\n  mAP50:    {coco_eval.stats[1]*100:.2f}%")
    print(f"  mAP50-95: {coco_eval.stats[0]*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="pascal", choices=["pascal", "ade20k234"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()
    data_root = args.data_root
    if not data_root:
        repo = Path(__file__).resolve().parent.parent.parent
        data_root = str(repo / "data" / ("ADE20KPart234" if args.dataset == "ade20k234" else "pascal_part"))
    evaluate(data_root, args.checkpoint, dataset_name=args.dataset, split=args.split, score_thresh=args.score_thresh)
