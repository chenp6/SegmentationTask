"""
COCO dataset for ConvNeXt prototype-based instance segmentation.

Returns per-image:
    image:   [3, H, W] float32 normalized
    targets: dict with boxes [N,4], labels [N], masks [N,H,W]
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import albumentations as A
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_coco(ann_path: str):
    with open(ann_path) as f:
        return json.load(f)


def polygons_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        return np.zeros((height, width), dtype=np.uint8)
    return coco_mask.decode(rle).astype(np.uint8)


def build_id_remap(categories):
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    return {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}


def get_train_transforms(image_size: int = 512):
    return A.Compose([
        A.SmallestMaxSize(max_size=image_size + 64),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8),
        A.GaussNoise(p=0.2),
    ], bbox_params=A.BboxParams(
        format="coco", label_fields=["category_ids"],
        min_area=64, min_visibility=0.3, clip=True,
    ))


def get_val_transforms(image_size: int = 512):
    return A.Compose([
        A.SmallestMaxSize(max_size=image_size),
        A.CenterCrop(height=image_size, width=image_size),
    ], bbox_params=A.BboxParams(
        format="coco", label_fields=["category_ids"],
        min_area=64, min_visibility=0.3, clip=True,
    ))


class HospitalCOCODataset(Dataset):
    """COCO dataset returning images + targets (boxes, labels, masks)."""

    def __init__(self, data_dir, split="train", image_size=512, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        ann_path = self.data_dir / split / "_annotations.coco.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation not found: {ann_path}")

        coco = load_coco(str(ann_path))
        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = coco["categories"]
        self.id_remap = build_id_remap(self.categories)
        self.num_classes = len(self.categories)

        self.ann_by_image = {}
        for ann in coco.get("annotations", []):
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

        self.image_ids = [
            img_id for img_id in self.images if img_id in self.ann_by_image
        ]

        self.transforms = (
            get_train_transforms(image_size) if augment
            else get_val_transforms(image_size)
        )

        print(f"[{split}] {len(self.image_ids)} images, "
              f"{self.num_classes} classes: {[c['name'] for c in self.categories]}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        anns = self.ann_by_image[img_id]

        img_path = self.data_dir / self.split / img_info["file_name"]
        image = np.array(Image.open(str(img_path)).convert("RGB"))
        h, w = image.shape[:2]

        # Collect masks, bboxes, labels
        masks, bboxes, labels = [], [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            bbox = ann.get("bbox", None)
            seg = ann.get("segmentation", [])
            if not seg or not bbox:
                continue
            mask = polygons_to_binary_mask(seg, h, w)
            if mask.sum() == 0:
                continue
            # COCO bbox: [x, y, w, h]
            bx, by, bw, bh = bbox
            if bw < 1 or bh < 1:
                continue
            masks.append(mask)
            bboxes.append([bx, by, bw, bh])
            labels.append(self.id_remap[ann["category_id"]])

        if not masks:
            masks = [np.zeros((h, w), dtype=np.uint8)]
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

        # Augmentation (handles bboxes + masks together)
        transformed = self.transforms(
            image=image, masks=masks, bboxes=bboxes, category_ids=labels
        )
        image_aug = transformed["image"]
        masks_aug = transformed["masks"]
        bboxes_aug = transformed["bboxes"]
        labels_aug = transformed["category_ids"]

        # Filter empty
        final_masks, final_bboxes, final_labels = [], [], []
        for m, b, l in zip(masks_aug, bboxes_aug, labels_aug):
            if m.sum() > 0 and b[2] > 0 and b[3] > 0:
                final_masks.append(m)
                final_bboxes.append(b)
                final_labels.append(l)

        if not final_masks:
            final_masks = [np.zeros(image_aug.shape[:2], dtype=np.uint8)]
            final_bboxes = [[0, 0, 1, 1]]
            final_labels = [0]

        # Normalize image
        image_float = image_aug.astype(np.float32) / 255.0
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)

        # Convert targets
        # Boxes: COCO [x,y,w,h] → [x1,y1,x2,y2] normalized to [0,1]
        img_h, img_w = image_aug.shape[:2]
        boxes_xyxy = []
        for bx, by, bw, bh in final_bboxes:
            boxes_xyxy.append([
                bx / img_w, by / img_h,
                (bx + bw) / img_w, (by + bh) / img_h
            ])

        return {
            "image": image_tensor,
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),  # [N, 4] normalized xyxy
            "labels": torch.tensor(final_labels, dtype=torch.long),
            "masks": torch.stack([torch.from_numpy(m).float() for m in final_masks]),
            "image_id": img_id,
        }


def collate_fn(batch):
    """Stack images, keep targets as list (variable number of instances)."""
    images = torch.stack([item["image"] for item in batch])
    targets = [{
        "boxes": item["boxes"],
        "labels": item["labels"],
        "masks": item["masks"],
    } for item in batch]
    return images, targets
