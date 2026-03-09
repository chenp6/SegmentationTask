"""
Evaluate SAM3 query-based instance segmentation on the hospital test set (COCO mAP50).

Query-based inference: threshold query scores, upsample masks to original size.
No NMS, no clustering, no offset voting — just score threshold.

Usage:
    python -m scripts.sam3_seg.evaluate
    python -m scripts.sam3_seg.evaluate --checkpoint output/sam3/best_model
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

from .config import DataConfig, ModelConfig
from .dataset import load_coco, IMAGENET_MEAN, IMAGENET_STD
from .segmentation_model import SAM3SegModel


def mask_to_rle(binary_mask):
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def extract_instances(pred_logits, pred_masks, num_classes, score_thresh=0.3, min_pixels=50):
    """
    Extract instances from query predictions — O(N_queries), trivially fast.

    Args:
        pred_logits: [N, C+1] class logits per query
        pred_masks:  [N, H_dec, W_dec] mask logits per query
        num_classes: number of classes (including background, NOT including no-object)

    Returns:
        list of {"class_idx": int, "mask": Tensor[H,W], "score": float}
    """
    no_obj_class = num_classes  # "no object" = last class

    # Softmax over classes
    probs = pred_logits.softmax(-1)  # [N, C+1]

    # Best class per query (excluding no-object)
    cls_probs = probs[:, :num_classes]  # [N, C]
    scores, cls_ids = cls_probs.max(-1)  # [N], [N]

    # Filter by score
    keep = (scores > score_thresh) & (cls_ids > 0)  # skip background (class 0)

    instances = []
    for idx in keep.nonzero(as_tuple=True)[0]:
        mask_logit = pred_masks[idx]  # [H_dec, W_dec]
        mask_bin = (mask_logit.sigmoid() > 0.5)

        if mask_bin.sum() < min_pixels:
            continue

        instances.append({
            "class_idx": cls_ids[idx].item() - 1,  # 0-indexed for COCO (subtract bg)
            "mask": mask_bin,  # still a tensor, will upsample later
            "score": scores[idx].item(),
        })

    return instances


def load_model(checkpoint_dir, device):
    ckpt_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "training_state.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SAM3SegModel.build(
        model_name=ckpt.get("model_name", "facebook/sam3-hiera-large"),
        num_classes=ckpt["num_classes"],
        use_adapters=ckpt.get("use_adapters", True),
        adapter_dim=ckpt.get("adapter_dim", 64),
        image_size=ckpt.get("image_size", 1024),
        decoder_type=ckpt.get("decoder_type", "query"),
        num_queries=ckpt.get("num_queries", 100),
        decoder_layers=ckpt.get("decoder_layers", 3),
        d_model=ckpt.get("d_model", 256),
        nhead=ckpt.get("nhead", 8),
        dim_ff=ckpt.get("dim_ff", 1024),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def preprocess_image(pil_image, image_size):
    img = pil_image.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)


def evaluate(
    data_cfg=None, model_cfg=None, checkpoint=None,
    split="test", score_thresh=0.3,
):
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    checkpoint = checkpoint or os.path.join(model_cfg.output_dir, "best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {checkpoint}")
    model, ckpt = load_model(checkpoint, device)
    num_classes = ckpt["num_classes"]
    image_size = ckpt.get("image_size", 1024)

    ann_path = Path(data_cfg.data_root) / split / "_annotations.coco.json"
    coco_data = load_coco(str(ann_path))
    sorted_cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    idx_to_cat_id = {i: cat["id"] for i, cat in enumerate(sorted_cats)}
    print(f"Classes ({len(sorted_cats)}): {[c['name'] for c in sorted_cats]}")

    img_dir = Path(data_cfg.data_root) / split
    coco_gt = COCO(str(ann_path))
    coco_results = []

    with torch.no_grad():
        for img_info in tqdm(coco_data["images"], desc=f"Evaluating [{split}]"):
            img_path = img_dir / img_info["file_name"]
            pil_image = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_image.size

            input_t = preprocess_image(pil_image, image_size).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_t)

            # Extract instances (at decoder resolution)
            instances = extract_instances(
                outputs["pred_logits"][0],
                outputs["pred_masks"][0],
                num_classes, score_thresh=score_thresh,
            )

            for inst in instances:
                # Upsample mask to original image size
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
        print("No predictions — model may not be trained yet")
        return

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50 = coco_eval.stats[1]
    map50_95 = coco_eval.stats[0]
    print(f"\n{'='*50}")
    print(f"  mAP50:       {map50*100:.2f}%  (target: > 70%)")
    print(f"  mAP50:95:    {map50_95*100:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()

    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root

    evaluate(data_cfg=data_cfg, checkpoint=args.checkpoint,
             split=args.split, score_thresh=args.score_thresh)
