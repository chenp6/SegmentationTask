"""
Pascal-Part dataset for YOLO-style training: each part is one instance (box + label + mask).

Data layout:
  - data_root/VOCdevkit/VOC2010/JPEGImages/{id}.jpg
  - data_root/VOCdevkit/VOC2010/ImageSets/Main/{train,val}.txt  (official: trainval=10,103)
  - data_root/Annotations_Part/{id}.mat  (one .mat per trainval image; 10,103 total)

Official stats (Pascal-Part page): Training+validation = 10,103 images; testing = 9,637.
We use Main/train.txt and Main/val.txt so all loaded samples have part annotations.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """(H,W) binary -> coco [x, y, w, h]."""
    if mask.sum() == 0:
        return 0, 0, 1, 1
    ys, xs = np.where(mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    return x_min, y_min, x_max - x_min, y_max - y_min


def _load_part_label_map(mat_path: Path) -> Tuple[List[np.ndarray], List[str], int, int]:
    """
    Load .mat and return list of part masks and part names.
    Returns (masks, part_names, H, W). Masks are binary (H,W).
    """
    d = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    anno = d["anno"]
    objs = np.atleast_1d(anno.objects)
    masks = []
    part_names = []
    h, w = 0, 0
    for o in objs:
        for pa in np.atleast_1d(getattr(o, "parts", [])):
            m = np.atleast_2d(getattr(pa, "mask", None))
            if m is None or m.size == 0:
                continue
            m = (m > 0).astype(np.uint8)
            if m.sum() == 0:
                continue
            h, w = m.shape[0], m.shape[1]
            masks.append(m)
            part_names.append(getattr(pa, "part_name", "unknown"))
    return masks, part_names, h, w


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


def _build_part_index(annotation_dir: Path, image_ids: List[str]) -> Tuple[Dict[str, int], List[str]]:
    """Scan .mat files for part_name and build part_name -> index (0-based). Returns (mapping, sorted names)."""
    part_set = set()
    for iid in image_ids:
        mat_path = annotation_dir / f"{iid}.mat"
        if not mat_path.exists():
            continue
        try:
            masks, names, _, _ = _load_part_label_map(mat_path)
            for n in names:
                part_set.add(n)
        except Exception:
            continue
    sorted_parts = sorted(part_set)
    return {p: i for i, p in enumerate(sorted_parts)}, sorted_parts


class PascalPartDataset(Dataset):
    """
    Pascal-Part: each part is one instance. Returns same format as COCO dataset for YOLO:
      image: [3, H, W] tensor, normalized
      boxes: [N, 4] normalized xyxy
      labels: [N] part class index (0-based)
      masks: [N, H, W] float
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.voc = self.data_root / "VOCdevkit" / "VOC2010"
        self.ann_dir = self.data_root / "Annotations_Part"
        self.jpeg_dir = self.voc / "JPEGImages"

        # Official Pascal-Part uses Main split (trainval=10,103); Segmentation split is smaller (964 each).
        split_file = self.voc / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            split_file = self.voc / "ImageSets" / "Segmentation" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        all_ids = [line.strip() for line in open(split_file).readlines() if line.strip()]

        # Only keep images that have part annotations (.mat)
        self.image_ids = [iid for iid in all_ids if (self.ann_dir / f"{iid}.mat").exists()]

        # Build part name -> index from this split
        self.part_to_idx, self.part_names = _build_part_index(self.ann_dir, self.image_ids)
        self.num_classes = len(self.part_names)

        self.transforms = (
            get_train_transforms(image_size) if augment
            else get_val_transforms(image_size)
        )

        print(
            f"[parts_seg {split}] {len(self.image_ids)} images, "
            f"{self.num_classes} part classes (from {len(all_ids)} in split)"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        iid = self.image_ids[idx]
        img_path = self.jpeg_dir / f"{iid}.jpg"
        mat_path = self.ann_dir / f"{iid}.mat"

        image = np.array(Image.open(str(img_path)).convert("RGB"))
        masks, part_names, h, w = _load_part_label_map(mat_path)

        if not masks or h == 0 or w == 0:
            # Fallback: no parts
            masks = [np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)]
            part_names = [self.part_names[0]]
            bboxes = [[0, 0, 1, 1]]
            labels = [0]
        else:
            bboxes = [_mask_to_bbox(m) for m in masks]
            labels = [self.part_to_idx.get(n, 0) for n in part_names]

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

        # Filter empty after crop
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
        image_tensor = (image_tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)

        return {
            "image": image_tensor,
            "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32),
            "labels": torch.tensor(final_labels, dtype=torch.long),
            "masks": torch.stack([torch.from_numpy(m).float() for m in final_masks]),
            "image_id": iid,
        }


def collate_fn(batch):
    """Stack images, keep targets as list."""
    images = torch.stack([item["image"] for item in batch])
    targets = [
        {"boxes": item["boxes"], "labels": item["labels"], "masks": item["masks"]}
        for item in batch
    ]
    return images, targets
