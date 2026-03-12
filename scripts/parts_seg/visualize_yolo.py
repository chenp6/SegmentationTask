"""
Visualize SAM2+YOLO part segmentation predictions on val images.

Supports --dataset pascal | ade20k234. Saves overlaid images to output_dir.

Usage (from repo root):
  python -m scripts.parts_seg.visualize_yolo --checkpoint output/parts_seg_yolo/latest_model.pt --dataset ade20k234 --max_images 20
  python -m scripts.parts_seg.visualize_yolo --checkpoint output/parts_seg_yolo/latest_model.pt --dataset pascal --output_dir output/parts_seg_yolo/vis_val
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

try:
    from scripts.sam2_yolo_seg.evaluate_yolo import decode_predictions
    from scripts.sam2_yolo_seg.models.sam2_yolo_model import SAM2YOLO
    from scripts.convnext_seg.visualize import overlay_masks
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from scripts.sam2_yolo_seg.evaluate_yolo import decode_predictions
    from scripts.sam2_yolo_seg.models.sam2_yolo_model import SAM2YOLO
    from scripts.convnext_seg.visualize import overlay_masks

from .dataset import PascalPartDataset
from .dataset_ade20k234 import ADE20KPart234Dataset


def visualize(data_root, checkpoint_path, dataset_name="pascal", split="val", max_images=10, output_dir=None, score_thresh=0.3):
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

    idx_to_class = {c: f"part_{c}" for c in range(num_classes)}
    indices = list(range(len(val_dataset)))
    if max_images and len(indices) > max_images:
        indices = random.sample(indices, max_images)
    output_dir = Path(output_dir or f"output/parts_seg_yolo/visualizations/{dataset_name}_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Visualizing {len(indices)} images -> {output_dir}")

    with torch.no_grad():
        for i in tqdm(indices, desc="Visualizing"):
            sample = val_dataset[i]
            image_t = sample["image"]
            H, W = sample["masks"].shape[1], sample["masks"].shape[2]
            # Reconstruct RGB for display (denorm)
            img_np = (image_t.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            image_batch = image_t.unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                outputs = model(image_batch)
            instances = decode_predictions(
                outputs, model.strides, image_size, num_classes,
                score_thresh=score_thresh,
            )
            for inst in instances:
                mask_t = inst["mask"].unsqueeze(0).unsqueeze(0).float()
                mask_resized = F.interpolate(mask_t, (H, W), mode="bilinear", align_corners=False)
                inst["mask"] = (mask_resized.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            if instances:
                vis = overlay_masks(pil_image, instances, idx_to_class)
            else:
                vis = pil_image.copy()
                ImageDraw.Draw(vis).text((10, 10), "No detections", fill=(255, 0, 0))
            vis.save(output_dir / f"image_{i:05d}_pred.jpg", quality=92)
    print(f"Done. Images saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="pascal", choices=["pascal", "ade20k234"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()
    data_root = args.data_root
    if not data_root:
        repo = Path(__file__).resolve().parent.parent.parent
        data_root = str(repo / "data" / ("ADE20KPart234" if args.dataset == "ade20k234" else "pascal_part"))
    visualize(
        data_root, args.checkpoint, dataset_name=args.dataset, split=args.split,
        max_images=args.max_images, output_dir=args.output_dir, score_thresh=args.score_thresh,
    )
