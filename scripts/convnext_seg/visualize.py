"""
Visualize ConvNeXt prototype instance segmentation predictions.

Usage:
    python -m scripts.convnext_seg.visualize --split test --max_images 20
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .config import DataConfig, ModelConfig
from .dataset import load_coco, IMAGENET_MEAN, IMAGENET_STD
from .evaluate import load_model, preprocess_image, decode_predictions


PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
    (180, 180, 180), (200, 150, 100), (100, 200, 150), (150, 100, 200),
    (50, 200, 200), (200, 50, 100), (100, 50, 200), (200, 100, 50),
]


def overlay_masks(image, instances, idx_to_class, alpha=0.45):
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for inst in instances:
        cls_idx = inst["class_idx"]
        score = inst["score"]
        mask = inst["mask"]
        if not mask.any():
            continue

        name = idx_to_class.get(cls_idx, f"cls{cls_idx}")
        color = PALETTE[cls_idx % len(PALETTE)]

        mask_img = Image.fromarray((mask * int(alpha * 255)).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", img.size, color + (180,))
        overlay.paste(color_layer, mask=mask_img)

        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        draw.text((cx, cy), f"{name} {score:.2f}", fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")


def visualize(data_cfg=None, model_cfg=None, checkpoint=None,
              split="test", max_images=None, output_dir=None, score_thresh=0.3):
    data_cfg = data_cfg or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    checkpoint = checkpoint or os.path.join(model_cfg.output_dir, "best_model")
    output_dir = output_dir or os.path.join(model_cfg.output_dir, "visualizations", split)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(checkpoint, device)
    num_classes = ckpt["num_classes"]
    image_size = ckpt.get("image_size", 512)

    ann_path = Path(data_cfg.data_root) / split / "_annotations.coco.json"
    coco_data = load_coco(str(ann_path))
    sorted_cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    idx_to_class = {i: cat["name"] for i, cat in enumerate(sorted_cats)}

    images = coco_data["images"]
    if max_images:
        images = random.sample(images, min(max_images, len(images)))

    print(f"Visualizing {len(images)} images → {output_dir}")

    with torch.no_grad():
        for img_info in tqdm(images, desc="Visualizing"):
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
                inst["mask"] = (mask_orig.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            if instances:
                vis = overlay_masks(pil_image, instances, idx_to_class)
            else:
                vis = pil_image.copy()
                ImageDraw.Draw(vis).text((10, 10), "No detections", fill=(255, 0, 0))

            stem = Path(img_info["file_name"]).stem
            vis.save(os.path.join(output_dir, f"{stem}_pred.jpg"), quality=92)

    print(f"Done. Images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()

    visualize(checkpoint=args.checkpoint, split=args.split,
              max_images=args.max_images, output_dir=args.output_dir,
              score_thresh=args.score_thresh)
