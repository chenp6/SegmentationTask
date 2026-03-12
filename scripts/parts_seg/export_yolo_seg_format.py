"""
Export ADE20KPart234 (or Pascal-Part) to Ultralytics YOLO segmentation format.

Output layout:
  out_dir/
    images/train/, images/val/
    labels/train/, labels/val/   (.txt per image: "class_id x1 y1 x2 y2 ..." normalized 0-1)
    labels/train/*.hierarchy.json  (optional) one entry per line in .txt: {"object_id", "part_class_id"}
  data.yaml (generated; includes object_names when hierarchy is exported)

Training with YOLOv11-seg checkpoint uses the .txt as usual; hierarchy sidecars are for
evaluation, analysis, or two-stage models that need object→part links.

Requires: opencv-python, pycocotools (for hierarchy: decode RLE object masks).

Usage (from repo root):
  python -m scripts.parts_seg.export_yolo_seg_format --data_root data/ADE20KPart234 --output_dir data/ADE20KPart234_yolo_seg
  python -m scripts.parts_seg.export_yolo_seg_format --data_root data/ADE20KPart234 --output_dir data/ADE20KPart234_yolo_seg --preserve_hierarchy
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import cv2
except ImportError:
    cv2 = None
try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

# Reuse part index and mask extraction from dataset
PART_IGNORE = (0, 65535)


def _part_png_to_masks_labels(png: np.ndarray):
    masks, labels = [], []
    part_ids = np.unique(png)
    for pid in part_ids:
        if pid in PART_IGNORE:
            continue
        binary = (png == pid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        labeled, num = ndimage.label(binary)
        for i in range(1, num + 1):
            m = (labeled == i).astype(np.uint8)
            if m.sum() < 16:
                continue
            masks.append(m)
            labels.append(int(pid))
    return masks, labels


def _mask_to_polygon_normalized(mask: np.ndarray, w: int, h: int, min_points: int = 3):
    """Convert binary mask to normalized polygon (0-1) for YOLO-seg. Returns list of (x,y) pairs or None."""
    if mask.sum() < 10:
        return None
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < min_points:
        return None
    points = contour.reshape(-1, 2)
    xs = (points[:, 0].astype(np.float64) / w).clip(0, 1)
    ys = (points[:, 1].astype(np.float64) / h).clip(0, 1)
    if len(xs) > 500:
        step = max(1, len(xs) // 500)
        xs = xs[::step]
        ys = ys[::step]
    return np.stack([xs, ys], axis=1).flatten().tolist()


def _decode_rle(seg: dict, h: int, w: int) -> np.ndarray:
    """Decode COCO RLE to binary mask (H,W)."""
    if coco_mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(seg.get("counts"), bytes):
        seg = {**seg, "counts": seg["counts"].decode("utf-8")}
    return coco_mask.decode(seg).astype(np.uint8)


def _assign_object_ids_to_parts(
    part_masks: list,
    part_ids_list: list,
    image_id: str,
    anns_by_image: dict,
    height: int,
    width: int,
) -> list:
    """
    For each part instance (mask), assign object_id from the COCO annotation that overlaps most.
    Returns list of int (object_id) same length as part_masks; -1 if no match or no annotations.
    """
    if not part_masks or image_id not in anns_by_image or coco_mask is None:
        return [-1] * len(part_masks)
    anns = anns_by_image[image_id]
    object_masks = []
    object_ids = []
    for a in anns:
        seg = a.get("segmentation")
        if not seg:
            continue
        size = seg.get("size", [height, width])
        oh, ow = size[0], size[1]
        m = _decode_rle(seg, oh, ow)
        if m.shape[0] != height or m.shape[1] != width:
            m = np.array(Image.fromarray(m).resize((width, height), resample=Image.NEAREST))
        object_masks.append((m > 0).astype(np.uint8))
        object_ids.append(a["category_id"])
    if not object_masks:
        return [-1] * len(part_masks)
    out = []
    for pm in part_masks:
        best_id = -1
        best_iou = 0.0
        inter_p = pm.sum()
        if inter_p == 0:
            out.append(-1)
            continue
        for om, oid in zip(object_masks, object_ids):
            inter = (pm.astype(np.uint8) * om).sum()
            union = (pm.astype(np.uint8) + om).clip(0, 1).sum()
            if union <= 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_id = oid
        out.append(best_id)
    return out


def export_ade20k234(
    data_root: Path,
    output_dir: Path,
    max_images: int | None = None,
    preserve_hierarchy: bool = False,
):
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    for split, split_name in [("train", "training"), ("val", "validation")]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    with open(data_root / "ade20k_instance_train.json") as f:
        coco_train = json.load(f)
    with open(data_root / "ade20k_instance_val.json") as f:
        coco_val = json.load(f)

    part_ids = set()
    for a in coco_train.get("annotations", []) + coco_val.get("annotations", []):
        for pid in a.get("part_category_id", []):
            if pid not in PART_IGNORE and pid != 65535:
                part_ids.add(int(pid))
    part_list = sorted(part_ids)
    part_to_idx = {p: i for i, p in enumerate(part_list)}
    num_classes = len(part_list)

    # Object categories for hierarchy (id -> name)
    object_cats = {}
    for c in coco_train.get("categories", []) + coco_val.get("categories", []):
        object_cats[c["id"]] = c.get("name", f"obj_{c['id']}")

    def anns_by_image_id(coco: dict) -> dict:
        by_img = {}
        for a in coco.get("annotations", []):
            iid = a["image_id"]
            if iid not in by_img:
                by_img[iid] = []
            by_img[iid].append(a)
        return by_img

    def process_split(coco: dict, split: str, img_subdir: str):
        images = coco["images"]
        if max_images and len(images) > max_images:
            images = images[:max_images]
        anns_by_img = anns_by_image_id(coco)
        img_dir = data_root / "images" / img_subdir
        part_dir = data_root / "annotations_detectron2_part" / img_subdir
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        for img_info in images:
            file_name = img_info["file_name"]
            image_id = img_info["id"]
            base = Path(file_name).stem
            img_path = img_dir / file_name
            part_path = part_dir / f"{base}.png"
            if not img_path.is_file() or not part_path.is_file():
                continue
            image = np.array(Image.open(img_path))
            h, w = image.shape[:2]
            png = np.array(Image.open(part_path))
            if png.ndim == 3:
                png = png[:, :, 0]
            if png.shape[0] != h or png.shape[1] != w:
                png = np.array(
                    Image.fromarray(png.astype(np.uint16)).resize((w, h), resample=Image.NEAREST)
                )
            masks, part_ids_list = _part_png_to_masks_labels(png)
            if preserve_hierarchy and coco_mask is not None:
                object_ids = _assign_object_ids_to_parts(
                    masks, part_ids_list, image_id, anns_by_img, h, w
                )
            else:
                object_ids = [-1] * len(masks)
            lines = []
            hierarchy_list = []
            for m, pid, oid in zip(masks, part_ids_list, object_ids):
                if pid not in part_to_idx:
                    continue
                poly = _mask_to_polygon_normalized(m, w, h)
                if poly is None or len(poly) < 6:
                    continue
                cls_id = part_to_idx[pid]
                line = f"{cls_id} " + " ".join(f"{x:.6f}" for x in poly)
                lines.append(line)
                if preserve_hierarchy:
                    hierarchy_list.append({"object_id": int(oid), "part_class_id": cls_id})
            out_img_path = out_img_dir / file_name
            if not out_img_path.exists():
                import shutil
                shutil.copy2(img_path, out_img_path)
            out_lbl_path = out_lbl_dir / f"{base}.txt"
            with open(out_lbl_path, "w") as f:
                f.write("\n".join(lines))
            if preserve_hierarchy and hierarchy_list:
                with open(out_lbl_dir / f"{base}.hierarchy.json", "w") as f:
                    json.dump(hierarchy_list, f, indent=0)
        return len(images)

    n_train = process_split(coco_train, "train", "training")
    n_val = process_split(coco_val, "val", "validation")

    names = [f"part_{i}" for i in range(num_classes)]
    with open(output_dir / "data.yaml", "w") as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {num_classes}\n")
        f.write(f"names: {names}\n")
        if preserve_hierarchy and object_cats:
            f.write("# object_id -> name; .hierarchy.json per image has one entry per .txt line\n")
            f.write(f"object_names: {json.dumps(object_cats)}\n")
    print(f"Exported {n_train} train, {n_val} val images, {num_classes} classes -> {output_dir}")
    if preserve_hierarchy:
        print("Hierarchy preserved: one .hierarchy.json per image (object_id per part instance).")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ade20k234")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--preserve_hierarchy",
        action="store_true",
        help="Write .hierarchy.json per image (object_id per part instance). Requires pycocotools.",
    )
    args = parser.parse_args()
    if cv2 is None:
        raise RuntimeError("opencv-python is required: pip install opencv-python")
    if args.preserve_hierarchy and coco_mask is None:
        raise RuntimeError("--preserve_hierarchy requires pycocotools: pip install pycocotools")
    output_dir = args.output_dir or (Path(args.data_root).parent / f"{Path(args.data_root).name}_yolo_seg")
    if args.dataset == "ade20k234":
        export_ade20k234(
            Path(args.data_root),
            Path(output_dir),
            args.max_images,
            preserve_hierarchy=args.preserve_hierarchy,
        )
    else:
        raise NotImplementedError(f"Export for dataset {args.dataset} not implemented. Use ade20k234.")


if __name__ == "__main__":
    main()
