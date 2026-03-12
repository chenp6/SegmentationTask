"""
PartImageNet++ from Hugging Face (lixiao20/PartImageNetPP) for YOLO-style training.

Loads annotation JSONs from HF; images are read from a local directory (ImageNet-style layout).
Dataset: 100k images, 3308 part categories, 406k part masks. COCO-style per-category JSONs.

Requires: pip install datasets huggingface_hub pycocotools
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as A
import numpy as np
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


def _decode_polygon(seg: list, height: int, width: int) -> np.ndarray:
    """Decode COCO polygon [[x,y,...], ...] to binary mask."""
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        coco_mask = None
    if coco_mask is None:
        return np.zeros((height, width), dtype=np.uint8)
    if not seg or not isinstance(seg[0], (list, tuple)):
        return np.zeros((height, width), dtype=np.uint8)
    rles = coco_mask.frPyObjects(seg, height, width)
    rle = coco_mask.merge(rles)
    return coco_mask.decode(rle).astype(np.uint8)


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
            format="coco", label_fields=["category_ids"], min_area=16, min_visibility=0.2, clip=True
        ),
    )


def get_val_transforms(image_size: int = 512):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ],
        bbox_params=A.BboxParams(
            format="coco", label_fields=["category_ids"], min_area=16, min_visibility=0.2, clip=True
        ),
    )


def _ensure_data_dir(
    data_dir: Path,
    repo_id: str = "lixiao20/PartImageNetPP",
    repo_type: str = "dataset",
    show_progress: bool = True,
) -> None:
    """Download PartImageNetPP annotation files from HF into data_dir (e.g. data/PartImageNetPP)."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError("PartImageNetPP requires: pip install huggingface_hub")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "json").mkdir(parents=True, exist_ok=True)

    all_files = list_repo_files(repo_id, repo_type=repo_type)
    to_download = [f for f in all_files if f.endswith(".json") and not f.startswith("demo/")]
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    iterator = tqdm(to_download, desc="PartImageNet++", unit="file") if (show_progress and tqdm) else to_download
    for rel_path in iterator:
        local_path = data_dir / rel_path
        if local_path.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path = hf_hub_download(repo_id, rel_path, repo_type=repo_type)
        with open(blob_path) as f:
            content = f.read()
        with open(local_path, "w") as f:
            f.write(content)


def _load_partimagenetpp_meta(
    data_dir: str | Path | None,
    cache_dir: str | None,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], set, int]:
    """
    Load PartImageNetPP annotations from data_dir if present, else download from HF.
    When data_dir is set, files are saved under data_dir so future loads use the data folder.
    Returns (samples, part_name_to_idx, discarded_basenames, n_val).
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise ImportError("PartImageNetPP requires: pip install huggingface_hub")

    repo_id = "lixiao20/PartImageNetPP"
    repo_type = "dataset"
    data_path = Path(data_dir) if data_dir else None

    def read_discarded():
        if data_path and (data_path / "discarded_data.json").exists():
            with open(data_path / "discarded_data.json") as f:
                return json.load(f)
        if data_path:
            _ensure_data_dir(data_path, repo_id, repo_type)
            with open(data_path / "discarded_data.json") as f:
                return json.load(f)
        path = hf_hub_download(repo_id, "discarded_data.json", repo_type=repo_type, cache_dir=cache_dir)
        with open(path) as f:
            return json.load(f)

    discarded_list = read_discarded()
    discarded_basenames = {str(x.get("file_name", "")).strip() for x in discarded_list if x.get("file_name")}

    if data_path and (data_path / "json").exists() and list((data_path / "json").glob("*.json")):
        json_files = sorted((data_path / "json").glob("*.json"))
        json_files = [str(f.relative_to(data_path)) for f in json_files]
        json_files = [f.replace("\\", "/") for f in json_files]
    else:
        if data_path:
            _ensure_data_dir(data_path, repo_id, repo_type)
            json_files = sorted(str(f.relative_to(data_path)).replace("\\", "/") for f in (data_path / "json").glob("*.json"))
        else:
            all_files = list_repo_files(repo_id, repo_type=repo_type)
            json_files = sorted(f for f in all_files if f.startswith("json/") and f.endswith(".json"))

    def open_json(rel_path: str):
        if data_path:
            local = data_path / rel_path
            if local.exists():
                with open(local) as f:
                    return json.load(f)
        if data_path:
            _ensure_data_dir(data_path, repo_id, repo_type)
            with open(data_path / rel_path) as f:
                return json.load(f)
        path = hf_hub_download(repo_id, rel_path, repo_type=repo_type, cache_dir=cache_dir)
        with open(path) as f:
            return json.load(f)

    # First pass: collect global part names
    part_names_set = set()
    for jf in json_files:
        data = open_json(jf)
        for c in data.get("categories", []):
            name = c.get("name")
            if name:
                part_names_set.add(name)
    part_names = sorted(part_names_set)
    part_name_to_idx = {n: i for i, n in enumerate(part_names)}

    # Second pass: build samples
    samples = []
    for jf in json_files:
        data = open_json(jf)
        local_cat_id_to_global = {}
        for c in data.get("categories", []):
            name = c.get("name")
            if name and name in part_name_to_idx:
                local_cat_id_to_global[c["id"]] = part_name_to_idx[name]
        image_id_to_info = {img["id"]: img for img in data.get("images", [])}
        by_image: Dict[int, List[Dict]] = {}
        for ann in data.get("annotations", []):
            image_id = ann.get("image_id")
            img_info = image_id_to_info.get(image_id)
            if not img_info:
                continue
            file_name = img_info.get("file_name", "")
            if not file_name:
                continue
            basename = os.path.basename(file_name)
            if basename in discarded_basenames:
                continue
            global_cid = local_cat_id_to_global.get(ann.get("category_id"))
            if global_cid is None:
                continue
            seg = ann.get("segmentation")
            if not seg:
                continue
            if image_id not in by_image:
                by_image[image_id] = []
            by_image[image_id].append({"segmentation": seg, "category_id": global_cid})
        for image_id, anns in by_image.items():
            img_info = image_id_to_info[image_id]
            file_name = img_info.get("file_name", "")
            samples.append({
                "image_path": file_name,
                "width": img_info.get("width", 640),
                "height": img_info.get("height", 480),
                "annotations": anns,
            })

    random.Random(seed).shuffle(samples)
    n_val = max(0, int(len(samples) * val_ratio))
    return samples, part_name_to_idx, discarded_basenames, n_val


# Default annotation dir: save/load PartImageNetPP JSONs under repo data folder
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "PartImageNetPP"


class PartImageNetPPDataset(Dataset):
    """
    PartImageNet++ from Hugging Face (lixiao20/PartImageNetPP).
    Annotations are loaded from data_dir (default: data/PartImageNetPP); if missing, downloaded from HF and saved there.
    Images are read from image_root (local). image_root must contain category subfolders, e.g. image_root/n01440764/n01440764_14524.JPEG.

    Returns same format as PascalPartDataset: image, boxes (normalized xyxy), labels, masks.
    """

    def __init__(
        self,
        image_root: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
        data_dir: str | Path | None = None,
        cache_dir: str | None = None,
        max_images: int | None = None,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.image_root = Path(image_root)
        self.split = split
        self.image_size = image_size
        data_dir = data_dir or _DEFAULT_DATA_DIR
        self._samples, self._part_to_idx, _, n_val = _load_partimagenetpp_meta(
            data_dir, cache_dir, val_ratio, seed
        )
        self.num_classes = len(self._part_to_idx)

        # Split: samples are already shuffled; first (len - n_val) are train, last n_val are val
        if split == "validation" or split == "val":
            self._samples = self._samples[-n_val:] if n_val else self._samples
        else:
            self._samples = self._samples[: max(0, len(self._samples) - n_val)]

        if max_images is not None and len(self._samples) > max_images:
            self._samples = self._samples[:max_images]

        self.transforms = get_train_transforms(image_size) if augment else get_val_transforms(image_size)
        print(
            f"[PartImageNetPP {split}] {len(self._samples)} images, "
            f"{self.num_classes} part classes (image_root={self.image_root})"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self._samples[idx]
        image_path = self.image_root / sample["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"PartImageNet++ image not found: {image_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]
        sample_w = sample.get("width") or w
        sample_h = sample.get("height") or h

        masks, bboxes, labels = [], [], []
        for a in sample["annotations"]:
            seg = a.get("segmentation")
            if not seg:
                continue
            mask = _decode_polygon(seg, sample_h, sample_w)
            if mask.sum() == 0:
                continue
            if mask.shape[0] != h or mask.shape[1] != w:
                from PIL import Image as PImage
                mask = np.array(
                    PImage.fromarray(mask).resize((w, h), resample=PImage.NEAREST)
                )
            cid = a.get("category_id", 0)
            masks.append(mask)
            bboxes.append(_mask_to_bbox(mask))
            labels.append(cid)

        if not masks:
            masks = [np.zeros((h, w), dtype=np.uint8)]
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

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
        image_tensor = (image_tensor - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)

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
