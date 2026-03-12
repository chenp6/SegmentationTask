"""
ADE20K Part (234 part categories) for YOLO-style training.

Data layout (data_root = data/ADE20KPart234):
  - images/training/*.jpg, images/validation/*.jpg
  - annotations_detectron2_part/training/*.png, annotations_detectron2_part/validation/*.png
    (16-bit PNG: pixel value = part category id; 0 and 65535 = background/unlabeled)
  - ade20k_instance_train.json, ade20k_instance_val.json (COCO-style; used for image list)

Each part instance from the part PNG (per connected component per part id) is returned as
box + label + mask, same format as PascalPartDataset.
"""
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Background / unlabeled in part PNG
PART_IGNORE = (0, 65535)


def _mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """(H,W) binary -> coco [x, y, w, h]."""
    if mask.sum() == 0:
        return 0, 0, 1, 1
    ys, xs = np.where(mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    return x_min, y_min, x_max - x_min, y_max - y_min


def _part_png_to_masks_labels(png: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
    """
    Convert part PNG (H, W) uint16 to list of instance masks and part labels.
    Each connected component of each part_id becomes one instance.
    """
    masks = []
    labels = []
    part_ids = np.unique(png)
    for pid in part_ids:
        if pid in PART_IGNORE:
            continue
        binary = (png == pid).astype(np.uint8)
        if binary.sum() == 0:
            continue
        # Connected components: each component = one instance
        labeled, num = ndimage.label(binary)
        for i in range(1, num + 1):
            m = (labeled == i).astype(np.uint8)
            if m.sum() < 16:  # min area
                continue
            masks.append(m)
            labels.append(int(pid))
    return masks, labels


def get_train_transforms(image_size: int = 512):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size + 64),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8),
            A.GaussNoise(p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=16,
            min_visibility=0.2,
            clip=True,
        ),
    )


def get_val_transforms(image_size: int = 512):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=16,
            min_visibility=0.2,
            clip=True,
        ),
    )


class ADE20KPart234Dataset(Dataset):
    """
    ADE20K Part (234 categories) from data/ADE20KPart234.
    Returns same format as PascalPartDataset: image, boxes (normalized xyxy), labels, masks.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
        max_images: int | None = None,
    ):
        data_root = Path(data_root)
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.img_dir = data_root / "images" / ("training" if split == "train" else "validation")
        self.part_dir = data_root / "annotations_detectron2_part" / (
            "training" if split == "train" else "validation"
        )
        json_name = "ade20k_instance_train.json" if split == "train" else "ade20k_instance_val.json"
        with open(data_root / json_name) as f:
            import json
            coco = json.load(f)
        self.images = coco["images"]
        if max_images is not None and len(self.images) > max_images:
            self.images = self.images[:max_images]
        # Build part id -> 0-based index from all part PNGs (or from JSON part_category_id)
        self._part_to_idx = self._build_part_index(coco)
        self.num_classes = len(self._part_to_idx)
        self.transforms = get_train_transforms(image_size) if augment else get_val_transforms(image_size)
        print(
            f"[ADE20KPart234 {split}] {len(self.images)} images, "
            f"{self.num_classes} part classes (data_root={data_root})"
        )

    def _build_part_index(self, coco: dict) -> dict:
        part_ids = set()
        for a in coco.get("annotations", []):
            for pid in a.get("part_category_id", []):
                if pid in PART_IGNORE or pid == 65535:
                    continue
                part_ids.add(int(pid))
        part_list = sorted(part_ids)
        return {p: i for i, p in enumerate(part_list)}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img_info = self.images[idx]
        file_name = img_info["file_name"]
        base = Path(file_name).stem
        image_path = self.img_dir / file_name
        part_path = self.part_dir / f"{base}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"ADE20KPart234 image not found: {image_path}")
        if not part_path.is_file():
            raise FileNotFoundError(f"ADE20KPart234 part PNG not found: {part_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        png = np.array(Image.open(part_path))
        if png.ndim == 3:
            png = png[:, :, 0]
        h, w = image.shape[:2]
        if png.shape[0] != h or png.shape[1] != w:
            png = np.array(
                Image.fromarray(png.astype(np.uint16)).resize((w, h), resample=Image.NEAREST)
            )
        masks, part_ids = _part_png_to_masks_labels(png)
        valid_masks, labels = [], []
        for m, pid in zip(masks, part_ids):
            if pid in self._part_to_idx:
                valid_masks.append(m)
                labels.append(self._part_to_idx[pid])
        masks = valid_masks
        if not masks:
            masks = [np.zeros((h, w), dtype=np.uint8)]
            bboxes = [[0, 0, 1, 1]]
            labels = [0]
        else:
            bboxes = [_mask_to_bbox(m) for m in masks]
        transformed = self.transforms(
            image=image,
            masks=masks,
            bboxes=bboxes,
            category_ids=labels,
        )
        image_aug = transformed["image"]
        masks_aug = transformed["masks"]
        bboxes_aug = transformed["bboxes"]
        labels_aug = transformed["category_ids"]
        final_masks, final_bboxes, final_labels = [], [], []
        for m, b, l in zip(masks_aug, bboxes_aug, labels_aug):
            if m.sum() > 0 and b[2] >= 1 and b[3] >= 1:
                final_masks.append(m)
                final_bboxes.append(b)
                final_labels.append(l)
        if not final_masks:
            final_masks = [np.zeros(image_aug.shape[:2], dtype=np.uint8)]
            final_bboxes = [[0, 0, 1, 1]]
            final_labels = [0]
        img_h, img_w = image_aug.shape[:2]
        boxes_xyxy = []
        for bx, by, bw, bh in final_bboxes:
            boxes_xyxy.append([
                bx / img_w, by / img_h,
                (bx + bw) / img_w, (by + bh) / img_h,
            ])
        image_tensor = torch.from_numpy(image_aug).permute(2, 0, 1).float() / 255.0
        image_tensor = (
            image_tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        ) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return {
            "image": image_tensor,
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.tensor(final_labels, dtype=torch.long),
            "masks": torch.stack([torch.from_numpy(m).float() for m in final_masks]),
            "image_id": idx,
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    targets = [
        {"boxes": item["boxes"], "labels": item["labels"], "masks": item["masks"]}
        for item in batch
    ]
    return images, targets
