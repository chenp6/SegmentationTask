from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset


def normalize_prompt_text(text: str) -> str:
    return " ".join(str(text).replace("_", " ").split())


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def segmentation_to_binary_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        return np.zeros((height, width), dtype=np.uint8)

    mask = coco_mask.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def resolve_split(data_root: Path, split: str) -> str:
    split = split.strip().lower()
    if (data_root / split / "_annotations.coco.json").exists():
        return split
    if split == "val" and (data_root / "valid" / "_annotations.coco.json").exists():
        return "valid"
    if split == "valid" and (data_root / "val" / "_annotations.coco.json").exists():
        return "val"
    raise FileNotFoundError(f"Cannot find split '{split}' under {data_root}")


def resolve_image_path(split_dir: Path, file_name: str) -> Path:
    p = split_dir / file_name
    if p.exists():
        return p
    p2 = split_dir / Path(file_name).name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Image not found under {split_dir}: {file_name}")


@dataclass
class InstanceSample:
    image_id: int
    annotation_id: int
    category_id: int
    category_name: str
    image_path: Path
    mask: np.ndarray


class COCOSam3LiteTextDataset(Dataset):
    """
    Per-instance COCO dataset for SAM3-LiteText.
    Each annotation becomes one sample with its own prompt text and mask.
    """

    def __init__(self, data_root: str | Path, split: str) -> None:
        self.data_root = Path(data_root)
        self.split = resolve_split(self.data_root, split)
        self.split_dir = self.data_root / self.split
        self.ann_path = self.split_dir / "_annotations.coco.json"

        coco = load_coco(self.ann_path)
        self.images = {int(img["id"]): img for img in coco.get("images", [])}
        self.categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
        self.category_id_to_name = {
            int(c["id"]): normalize_prompt_text(str(c.get("name", c["id"])))
            for c in self.categories
        }

        self.samples: list[InstanceSample] = []
        skipped = 0
        for ann in coco.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue

            image_id = int(ann.get("image_id", -1))
            img = self.images.get(image_id)
            if img is None:
                skipped += 1
                continue

            h = int(img.get("height", 0))
            w = int(img.get("width", 0))
            if h <= 0 or w <= 0:
                skipped += 1
                continue

            file_name = str(img.get("file_name", ""))
            try:
                image_path = resolve_image_path(self.split_dir, file_name)
            except FileNotFoundError:
                skipped += 1
                continue

            mask = segmentation_to_binary_mask(ann.get("segmentation"), h, w)
            if mask.sum() == 0:
                skipped += 1
                continue

            category_id = int(ann.get("category_id", -1))
            category_name = self.category_id_to_name.get(category_id, str(category_id))

            self.samples.append(
                InstanceSample(
                    image_id=image_id,
                    annotation_id=int(ann.get("id", -1)),
                    category_id=category_id,
                    category_name=category_name,
                    image_path=image_path,
                    mask=mask,
                )
            )

        print(
            f"[{self.split}] {len(self.samples)} samples, "
            f"{len(self.categories)} categories, skipped={skipped}"
        )
        if not self.samples:
            raise RuntimeError(f"No valid samples from {self.ann_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        image = Image.open(s.image_path).convert("RGB")
        return {
            "image": image,
            "text": s.category_name,
            "mask": s.mask,
            "image_id": s.image_id,
            "annotation_id": s.annotation_id,
            "category_id": s.category_id,
            "category_name": s.category_name,
            "file_name": s.image_path.name,
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [b["image"] for b in batch],
        "texts": [b["text"] for b in batch],
        "masks": [b["mask"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "annotation_ids": [b["annotation_id"] for b in batch],
        "category_ids": [b["category_id"] for b in batch],
        "category_names": [b["category_name"] for b in batch],
        "file_names": [b["file_name"] for b in batch],
    }
