"""
Reusable augmentation settings shared by training scripts.

Also supports exporting an offline augmented COCO dataset.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


@dataclass
class AugmentConfig:
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


PROFILES: dict[str, AugmentConfig] = {
    "default": AugmentConfig(),
    "medbin_safe": AugmentConfig(
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.15,
        degrees=5.0,
        translate=0.08,
        scale=0.15,
        shear=1.0,
        perspective=0.01,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.2,
        mixup=0.0,
    ),
    "medbin_strong_rotate": AugmentConfig(
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.15,
        degrees=180.0,
        translate=0.08,
        scale=0.15,
        shear=1.0,
        perspective=0.01,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.2,
        mixup=0.0,
    ),
}


def load_augment_from_json(path: str) -> dict[str, float]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    valid_keys = set(AugmentConfig().__dict__.keys())
    unknown_keys = set(payload.keys()) - valid_keys
    if unknown_keys:
        unknown_sorted = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown augmentation keys in {path}: {unknown_sorted}")

    merged = AugmentConfig().__dict__.copy()
    merged.update(payload)
    return {k: float(v) for k, v in merged.items()}


def load_augment_from_profile(profile_name: str) -> dict[str, float]:
    if profile_name not in PROFILES:
        names = ", ".join(sorted(PROFILES.keys()))
        raise ValueError(f"Unknown augment profile '{profile_name}'. Available: {names}")
    return PROFILES[profile_name].to_dict()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _segmentation_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        return np.zeros((height, width), dtype=np.uint8)

    decoded = coco_mask.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def _binary_mask_to_polygons(mask: np.ndarray, min_area: float = 1.0) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygon = points.astype(np.float32).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def _mask_to_bbox_xywh(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _build_transform(cfg: dict[str, float]) -> A.Compose:
    transforms: list[A.BasicTransform] = []

    # Geometric transform approximating YOLO settings.
    if (
        cfg["degrees"] > 0
        or cfg["translate"] > 0
        or cfg["scale"] > 0
        or cfg["shear"] > 0
    ):
        transforms.append(
            A.Affine(
                scale=(max(0.1, 1.0 - cfg["scale"]), 1.0 + cfg["scale"]),
                translate_percent={"x": (-cfg["translate"], cfg["translate"]), "y": (-cfg["translate"], cfg["translate"])},
                rotate=(-cfg["degrees"], cfg["degrees"]),
                shear={"x": (-cfg["shear"], cfg["shear"]), "y": (-cfg["shear"], cfg["shear"])},
                fit_output=False,
                p=1.0,
            )
        )

    if cfg["perspective"] > 0:
        transforms.append(A.Perspective(scale=(0.0, cfg["perspective"]), keep_size=True, p=1.0))

    if cfg["flipud"] > 0:
        transforms.append(A.VerticalFlip(p=cfg["flipud"]))
    if cfg["fliplr"] > 0:
        transforms.append(A.HorizontalFlip(p=cfg["fliplr"]))

    # Color transform approximating YOLO HSV params.
    if cfg["hsv_h"] > 0 or cfg["hsv_s"] > 0 or cfg["hsv_v"] > 0:
        transforms.append(
            A.ColorJitter(
                brightness=cfg["hsv_v"],
                contrast=0.0,
                saturation=cfg["hsv_s"],
                hue=cfg["hsv_h"],
                p=1.0,
            )
        )

    if not transforms:
        transforms.append(A.NoOp())
    return A.Compose(transforms)


def _resolve_image_path(split_dir: Path, file_name: str) -> Path:
    direct = split_dir / file_name
    if direct.exists():
        return direct
    by_name = split_dir / Path(file_name).name
    if by_name.exists():
        return by_name
    raise FileNotFoundError(f"Image not found under {split_dir}: {file_name}")


def _copy_original_split(coco: dict, src_split_dir: Path, dst_split_dir: Path) -> tuple[list[dict], list[dict]]:
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    new_images: list[dict] = []
    for img in coco.get("images", []):
        src_path = _resolve_image_path(src_split_dir, img["file_name"])
        dst_name = Path(img["file_name"]).name
        shutil.copy2(src_path, dst_split_dir / dst_name)

        img_out = copy.deepcopy(img)
        img_out["file_name"] = dst_name
        new_images.append(img_out)

    new_annotations = [copy.deepcopy(ann) for ann in coco.get("annotations", [])]
    return new_images, new_annotations


def export_augmented_coco_dataset(
    input_root: str,
    output_root: str,
    augment_values: dict[str, float],
    copies_per_image: int = 1,
    splits: tuple[str, ...] = ("train",),
    include_originals: bool = True,
    min_mask_area: int = 8,
) -> None:
    in_root = Path(input_root).resolve()
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if copies_per_image < 1:
        raise ValueError(f"copies_per_image must be >= 1, got {copies_per_image}")

    transform = _build_transform(augment_values)

    for split in ("train", "valid", "test"):
        src_split_dir = in_root / split
        src_ann_path = src_split_dir / "_annotations.coco.json"
        if not src_ann_path.exists():
            continue

        dst_split_dir = out_root / split
        coco = json.loads(src_ann_path.read_text(encoding="utf-8"))

        if include_originals:
            images_out, annotations_out = _copy_original_split(coco, src_split_dir, dst_split_dir)
        else:
            dst_split_dir.mkdir(parents=True, exist_ok=True)
            images_out, annotations_out = [], []

        next_image_id = max([int(i["id"]) for i in images_out], default=0) + 1
        next_ann_id = max([int(a["id"]) for a in annotations_out], default=0) + 1
        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        do_augment = split in splits
        aug_images = 0
        aug_annotations = 0

        for img in coco.get("images", []):
            src_path = _resolve_image_path(src_split_dir, img["file_name"])
            if src_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not do_augment:
                continue

            image = np.array(Image.open(src_path).convert("RGB"))
            h, w = image.shape[:2]
            image_anns = anns_by_image.get(int(img["id"]), [])

            masks: list[np.ndarray] = []
            mask_ann_sources: list[dict] = []
            for ann in image_anns:
                if ann.get("iscrowd", 0):
                    continue
                seg = ann.get("segmentation")
                if not seg:
                    continue
                mask = _segmentation_to_binary_mask(seg, h, w)
                if mask.sum() == 0:
                    continue
                masks.append(mask)
                mask_ann_sources.append(ann)

            if not masks:
                continue

            for copy_idx in range(copies_per_image):
                transformed = transform(image=image, masks=masks)
                aug_image = transformed["image"]
                aug_masks = transformed["masks"]

                suffix = src_path.suffix.lower() if src_path.suffix.lower() in IMAGE_EXTENSIONS else ".jpg"
                aug_name = f"{Path(img['file_name']).stem}__aug_{copy_idx:02d}{suffix}"
                aug_path = dst_split_dir / aug_name
                Image.fromarray(aug_image).save(aug_path)

                new_image_id = next_image_id
                next_image_id += 1
                images_out.append(
                    {
                        "id": new_image_id,
                        "width": int(aug_image.shape[1]),
                        "height": int(aug_image.shape[0]),
                        "file_name": aug_name,
                    }
                )
                aug_images += 1

                for src_ann, aug_mask in zip(mask_ann_sources, aug_masks):
                    area = int((aug_mask > 0).sum())
                    if area < min_mask_area:
                        continue

                    polygons = _binary_mask_to_polygons(aug_mask, min_area=1.0)
                    if not polygons:
                        continue

                    bbox = _mask_to_bbox_xywh(aug_mask)
                    if bbox is None:
                        continue

                    new_ann = copy.deepcopy(src_ann)
                    new_ann["id"] = next_ann_id
                    next_ann_id += 1
                    new_ann["image_id"] = new_image_id
                    new_ann["segmentation"] = polygons
                    new_ann["bbox"] = bbox
                    new_ann["area"] = float(area)
                    new_ann["iscrowd"] = 0
                    annotations_out.append(new_ann)
                    aug_annotations += 1

        out_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco.get("categories", []),
            "images": images_out,
            "annotations": annotations_out,
        }

        out_ann_path = dst_split_dir / "_annotations.coco.json"
        out_ann_path.write_text(json.dumps(out_coco, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{split}] output: {dst_split_dir}")
        print(f"  images: {len(images_out)}")
        print(f"  annotations: {len(annotations_out)}")
        if do_augment:
            print(f"  augmented images added: {aug_images}")
            print(f"  augmented annotations added: {aug_annotations}")


def _parse_splits(value: str) -> tuple[str, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    valid = {"train", "valid", "test"}
    invalid = [p for p in parts if p not in valid]
    if invalid:
        raise ValueError(f"Invalid splits: {invalid}. Allowed: train,valid,test")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export offline augmented COCO dataset.")
    parser.add_argument("--input-root", required=True, help="Input COCO dataset root")
    parser.add_argument("--output-root", required=True, help="Output dataset root, e.g. data/medbin_dataset_aug")
    parser.add_argument("--augment-config", default=None, help="Path to augmentation JSON config")
    parser.add_argument("--augment-profile", default=None, help="Named profile in this file")
    parser.add_argument("--copies-per-image", type=int, default=1, help="How many augmented copies per source image")
    parser.add_argument("--splits", default="train", help="Comma-separated splits to augment, e.g. train or train,valid")
    parser.add_argument("--no-originals", action="store_true", help="Do not copy original images/annotations into output")
    parser.add_argument("--min-mask-area", type=int, default=8, help="Minimum mask area to keep an instance")
    args = parser.parse_args()

    if args.augment_config and args.augment_profile:
        raise ValueError("Use either --augment-config or --augment-profile, not both.")

    if args.augment_config:
        augment_values = load_augment_from_json(args.augment_config)
    else:
        augment_values = load_augment_from_profile(args.augment_profile or "default")

    export_augmented_coco_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        augment_values=augment_values,
        copies_per_image=args.copies_per_image,
        splits=_parse_splits(args.splits),
        include_originals=not args.no_originals,
        min_mask_area=args.min_mask_area,
    )


if __name__ == "__main__":
    main()
