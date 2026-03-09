"""
COCO Dataset for SAM3-UNet segmentation.

Converts COCO polygon annotations into:
  - image:         float32 tensor [3, H, W], normalised (ImageNet mean/std)
  - semantic_mask: LongTensor [H, W] with 0=background, 1..N=classes
  - instance_info: list of (binary_mask, class_label) for eval compatibility

Adapted from mask2former_seg/dataset.py.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset

# ImageNet normalisation used by SAM3 / Hiera
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def load_coco(ann_path: str):
    with open(ann_path) as f:
        return json.load(f)


def polygons_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Convert COCO polygon(s) or RLE to a binary mask (H, W) uint8."""
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        return np.zeros((height, width), dtype=np.uint8)
    binary = coco_mask.decode(rle).astype(np.uint8)
    return binary


def build_id_remap(categories: List[dict]) -> Dict[int, int]:
    """Create mapping from COCO category_id → 0-indexed class id."""
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    return {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}


# --------------------------------------------------------------------------- #
# Augmentations                                                                #
# --------------------------------------------------------------------------- #

def get_train_transforms(image_size: int = 1024) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size + 64),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8
            ),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )


def get_val_transforms(image_size: int = 1024) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ]
    )


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class HospitalCOCOSegDataset(Dataset):
    """
    COCO-format dataset returning tensors ready for the SAM3-UNet model.

    Returns dict:
        image:          float32 [3, H, W]  (ImageNet normalised)
        semantic_mask:  int64   [H, W]     (0=bg, 1..N=classes)
        instance_masks: list of uint8 [H, W] binary masks
        class_labels:   list of int (0-indexed class ids, NOT +1)
        image_id:       int
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 1024,
        augment: bool = True,
    ):
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

        # Group annotations by image_id
        self.ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Only keep images with at least one annotation
        self.image_ids = [
            img_id for img_id in self.images if img_id in self.ann_by_image
        ]

        self.transforms = (
            get_train_transforms(image_size) if augment
            else get_val_transforms(image_size)
        )

        print(
            f"[{split}] {len(self.image_ids)} images, "
            f"{self.num_classes} classes: {[c['name'] for c in self.categories]}"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        anns = self.ann_by_image[img_id]

        # Load image
        img_path = self.data_dir / self.split / img_info["file_name"]
        image = np.array(Image.open(str(img_path)).convert("RGB"))
        h, w = image.shape[:2]

        # Build per-instance masks and labels
        masks = []
        class_labels = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            mask = polygons_to_binary_mask(seg, h, w)
            if mask.sum() == 0:
                continue
            masks.append(mask)
            class_labels.append(self.id_remap[ann["category_id"]])

        if not masks:
            masks = [np.zeros((h, w), dtype=np.uint8)]
            class_labels = [0]

        # Apply augmentations (supports multiple masks)
        transformed = self.transforms(image=image, masks=masks)
        image_aug = transformed["image"]     # (H, W, 3) uint8
        masks_aug = transformed["masks"]     # list of (H, W) uint8

        # Filter empty masks after augmentation
        final_masks, final_labels = [], []
        for m, l in zip(masks_aug, class_labels):
            if m.sum() > 0:
                final_masks.append(m)
                final_labels.append(l)

        if not final_masks:
            final_masks = [np.zeros(image_aug.shape[:2], dtype=np.uint8)]
            final_labels = [0]

        # Build semantic mask: 0=background, class+1 for each class
        # When instances of the same class overlap, later instances overwrite
        sem_h, sem_w = image_aug.shape[:2]
        semantic_mask = np.zeros((sem_h, sem_w), dtype=np.int64)
        for m, l in zip(final_masks, final_labels):
            semantic_mask[m > 0] = l + 1  # +1 so 0 stays as background

        # Normalise image → float32 tensor [3, H, W]
        image_float = image_aug.astype(np.float32) / 255.0
        image_float = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_float).permute(2, 0, 1)  # [3,H,W]

        semantic_tensor = torch.from_numpy(semantic_mask)  # [H, W] int64

        return {
            "image": image_tensor,
            "semantic_mask": semantic_tensor,
            "instance_masks": final_masks,      # list of np (H,W) uint8
            "class_labels": final_labels,        # list of int
            "image_id": img_id,
        }


def collate_fn(batch):
    """Simple collate: stack images and semantic masks, keep instance info as lists."""
    images = torch.stack([item["image"] for item in batch])
    semantic_masks = torch.stack([item["semantic_mask"] for item in batch])
    return {
        "images": images,
        "semantic_masks": semantic_masks,
        "instance_masks": [item["instance_masks"] for item in batch],
        "class_labels": [item["class_labels"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
    }
