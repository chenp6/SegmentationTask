"""
Evaluate ConvNeXt prototype instance segmentation (COCO mAP50).

Usage:
    python -m scripts.convnext_seg.evaluate
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torchvision.ops import nms

from .config import DataConfig, ModelConfig
from .dataset import load_coco, IMAGENET_MEAN, IMAGENET_STD
from .model import ConvNeXtSegModel


def mask_to_rle(binary_mask):
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def preprocess_image(pil_image, image_size):
    img = pil_image.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def decode_predictions(outputs, strides, image_size, num_classes,
                       score_thresh=0.3, nms_thresh=0.5, max_dets=100):
    """
    Decode FCOS outputs into instance predictions.

    Returns list of {class_idx, score, bbox_xyxy, mask}
    """
    protos = outputs["protos"][0]  # [K, H_p, W_p]
    device = protos.device
    all_scores, all_classes, all_boxes, all_coeffs = [], [], [], []

    for level_idx, level_out in enumerate(outputs["levels"]):
        stride = strides[level_idx]
        cls_logits = level_out["cls"][0]   # [C, H, W]
        bbox_pred = level_out["bbox"][0]    # [4, H, W]
        ctr_pred = level_out["centerness"][0, 0]  # [H, W]
        coeff = level_out["coeff"][0]       # [K, H, W]

        H, W = cls_logits.shape[1:]

        # Grid positions
        shifts_y = (torch.arange(H, device=device).float() + 0.5) * stride
        shifts_x = (torch.arange(W, device=device).float() + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        # Class scores * centerness
        cls_scores = cls_logits.sigmoid() * ctr_pred.sigmoid()  # [C, H, W]
        max_scores, max_cls = cls_scores.max(dim=0)  # [H, W]

        # Filter by score
        keep = max_scores > score_thresh
        if not keep.any():
            continue

        scores = max_scores[keep]
        classes = max_cls[keep]
        cx = shift_x[keep]
        cy = shift_y[keep]
        l, t, r, b = bbox_pred[0][keep], bbox_pred[1][keep], bbox_pred[2][keep], bbox_pred[3][keep]
        coefficients = coeff[:, keep].T  # [N, K]

        # Convert ltbr to xyxy boxes
        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b

        all_scores.append(scores)
        all_classes.append(classes)
        all_boxes.append(torch.stack([x1, y1, x2, y2], dim=1))
        all_coeffs.append(coefficients)

    if not all_scores:
        return []

    scores = torch.cat(all_scores)
    classes = torch.cat(all_classes)
    boxes = torch.cat(all_boxes)
    coeffs = torch.cat(all_coeffs)

    # NMS per class
    keep = nms(boxes, scores, nms_thresh)
    if len(keep) > max_dets:
        keep = keep[:max_dets]

    instances = []
    for idx in keep:
        coeff = coeffs[idx]  # [K]
        pred_mask = (coeff[:, None, None] * protos).sum(dim=0).sigmoid()  # [H_p, W_p]
        mask_bin = (pred_mask > 0.5)

        instances.append({
            "class_idx": classes[idx].item(),
            "score": scores[idx].item(),
            "bbox": boxes[idx].cpu(),
            "mask": mask_bin,
        })

    return instances


def load_model(checkpoint_dir, device):
    ckpt_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "training_state.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = ConvNeXtSegModel.build(
        backbone_name=ckpt.get("backbone", "convnext_small"),
        num_classes=ckpt["num_classes"],
        d_model=ckpt.get("d_model", 256),
        num_protos=ckpt.get("num_prototypes", 32),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def evaluate(data_cfg=None, model_cfg=None, checkpoint=None, split="test", score_thresh=0.3):
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    checkpoint = checkpoint or os.path.join(model_cfg.output_dir, "best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {checkpoint}")
    model, ckpt = load_model(checkpoint, device)
    num_classes = ckpt["num_classes"]
    image_size = ckpt.get("image_size", 512)

    ann_path = Path(data_cfg.data_root) / split / "_annotations.coco.json"
    coco_data = load_coco(str(ann_path))
    sorted_cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    idx_to_cat_id = {i: cat["id"] for i, cat in enumerate(sorted_cats)}

    coco_gt = COCO(str(ann_path))
    coco_results = []

    with torch.no_grad():
        for img_info in tqdm(coco_data["images"], desc=f"Evaluating [{split}]"):
            img_path = Path(data_cfg.data_root) / split / img_info["file_name"]
            pil_image = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_image.size

            input_t = preprocess_image(pil_image, image_size).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_t)

            instances = decode_predictions(
                outputs, model.strides, image_size, num_classes,
                score_thresh=score_thresh,
            )

            for inst in instances:
                mask_t = inst["mask"].unsqueeze(0).unsqueeze(0).float()
                mask_orig = F.interpolate(mask_t, (orig_h, orig_w), mode="bilinear", align_corners=False)
                mask_np = (mask_orig.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                cat_id = idx_to_cat_id.get(inst["class_idx"], -1)
                if cat_id == -1:
                    continue
                coco_results.append({
                    "image_id": img_info["id"],
                    "category_id": cat_id,
                    "segmentation": mask_to_rle(mask_np),
                    "score": inst["score"],
                })

    print(f"\nTotal predictions: {len(coco_results)}")
    if not coco_results:
        print("No predictions")
        return

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50 = coco_eval.stats[1]
    print(f"\n{'='*50}")
    print(f"  mAP50:    {map50*100:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()

    evaluate(checkpoint=args.checkpoint, split=args.split, score_thresh=args.score_thresh)
