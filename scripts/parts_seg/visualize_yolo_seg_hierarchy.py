"""
Visualize YOLOv11-seg hierarchy model (part segmentation) predictions on validation images.

Loads a trained checkpoint and runs prediction, then overlays masks with part class labels.
Uses the same data.yaml as training to get val image paths.

Usage (from repo root):
  python -m scripts.parts_seg.visualize_yolo_seg_hierarchy --checkpoint output/parts_yolo_seg_hierarchy/train/weights/best.pt --data_yaml data/ADE20KPart234_yolo_seg/data.yaml --max_images 20
  python -m scripts.parts_seg.visualize_yolo_seg_hierarchy --checkpoint output/parts_yolo_seg_hierarchy/train/weights/best.pt --data_yaml data/ADE20KPart234_yolo_seg/data.yaml --output_dir output/parts_yolo_seg_hierarchy/vis_val
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics.utils import YAML

from .model_yolo_seg_hierarchy import load_yolo_seg_hierarchy_from_checkpoint

try:
    from scripts.convnext_seg.visualize import overlay_masks
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from scripts.convnext_seg.visualize import overlay_masks


def results_to_instances(results, score_thresh=0.25):
    """Convert Ultralytics segment Results to list of {mask, class_idx, score} for overlay_masks."""
    instances = []
    if not results or len(results) == 0:
        return instances
    r = results[0]
    if r.masks is None or r.boxes is None:
        return instances
    masks = r.masks.data.cpu().numpy()  # (N, H, W)
    boxes = r.boxes
    cls = boxes.cls.cpu().numpy()  # (N,)
    conf = boxes.conf.cpu().numpy()  # (N,)
    for i in range(len(masks)):
        if conf[i] < score_thresh:
            continue
        mask = (masks[i] > 0.5).astype(np.uint8)
        if mask.sum() == 0:
            continue
        instances.append({
            "mask": mask,
            "class_idx": int(cls[i]),
            "score": float(conf[i]),
        })
    return instances


def main():
    parser = argparse.ArgumentParser(description="Visualize hierarchy YOLOv11-seg part predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt from hierarchy training (e.g. .../weights/last.pt)")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to data.yaml (same as training)")
    parser.add_argument("--split", type=str, default="val", help="Split to visualize (default: val)")
    parser.add_argument("--max_images", type=int, default=20, help="Max number of images to visualize (0 = all)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for overlay images")
    parser.add_argument("--score_thresh", type=float, default=0.25, help="Min confidence to show instance")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    data = YAML.load(data_yaml) if str(data_yaml).endswith((".yaml", ".yml")) else {}
    data = data or {}
    base_path = data.get("path")
    if base_path is None:
        base_path = data_yaml.parent
    base_path = Path(base_path)
    if not base_path.is_absolute():
        base_path = (data_yaml.parent / base_path).resolve()
    val_key = args.split if args.split in data else "val"
    val_rel = data.get(val_key, "images/val")
    val_dir = (base_path / val_rel) if not Path(val_rel).is_absolute() else Path(val_rel)
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Val image dir not found: {val_dir} (from data.yaml path={base_path}, {val_key}={val_rel})")

    image_files = sorted(val_dir.glob("*.*"))
    image_files = [f for f in image_files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    if not image_files:
        raise FileNotFoundError(f"No images in {val_dir}")

    if args.max_images > 0 and len(image_files) > args.max_images:
        image_files = random.sample(image_files, args.max_images)

    print(f"Loading hierarchy model from {args.checkpoint}")
    model, _ = load_yolo_seg_hierarchy_from_checkpoint(
        args.checkpoint,
        data_yaml=str(data_yaml),
        device=args.device or None,
    )
    nc = model.model.nc
    names = data.get("names")
    if isinstance(names, list):
        idx_to_class = {i: names[i] for i in range(min(nc, len(names)))}
        for i in range(len(idx_to_class), nc):
            idx_to_class[i] = f"part_{i}"
    elif isinstance(names, dict):
        idx_to_class = {int(k): v for k, v in names.items() if int(k) < nc}
        for i in range(nc):
            idx_to_class.setdefault(i, f"part_{i}")
    else:
        idx_to_class = {i: f"part_{i}" for i in range(nc)}

    output_dir = Path(args.output_dir or f"output/parts_yolo_seg_hierarchy/visualizations/{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Visualizing {len(image_files)} images -> {output_dir}")

    for path in tqdm(image_files, desc="Visualizing"):
        results = model.predict(
            source=str(path),
            conf=args.score_thresh,
            save=False,
            device=args.device or None,
            verbose=False,
        )
        if not results:
            continue
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        instances = results_to_instances(results, score_thresh=args.score_thresh)
        # Ensure each mask matches the image size for overlay (PIL requires same HxW)
        if instances:
            for inst in instances:
                m = inst["mask"]
                if m.shape != (orig_h, orig_w):
                    from PIL import Image as _PILImage

                    m_img = _PILImage.fromarray((m > 0).astype(np.uint8) * 255).resize(
                        (orig_w, orig_h), resample=_PILImage.NEAREST
                    )
                    inst["mask"] = (np.array(m_img) > 0).astype(np.uint8)
            vis = overlay_masks(img, instances, idx_to_class)
        else:
            from PIL import ImageDraw
            vis = img.copy()
            ImageDraw.Draw(vis).text((10, 10), "No detections", fill=(255, 0, 0))
        out_name = path.stem + "_pred.jpg"
        vis.save(output_dir / out_name, quality=92)

    print(f"Done. Images saved to {output_dir}")


if __name__ == "__main__":
    main()
