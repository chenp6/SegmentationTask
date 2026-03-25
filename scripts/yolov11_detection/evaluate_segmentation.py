"""
Evaluate YOLOv11 detection predictions as instance segmentation via SAM2 box prompts.

Workflow:
1. Run YOLOv11 object detection on images from a COCO segmentation dataset split
2. Convert each predicted bbox into a SAM2-prompted mask
3. Evaluate the predicted masks against COCO segmentation ground truth

Usage:
    python -m scripts.yolov11_detection.evaluate_segmentation \
      --model output/yolov11_detection/exp/weights/best.pt \
      --input-root data/hiod_sam2_seg \
      --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import YOLO

from scripts.object_detection_to_image_segmentaion.convert_coco_detection_to_segmentation import (
    SAM2BoxSegmenter,
    build_segmentation,
    expand_box_xyxy,
    load_coco,
    resolve_image_path,
)


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def box_xyxy_to_mask(
    box_xyxy: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(x1, min(x2, width))
    y2 = max(y1, min(y2, height))
    mask[y1:y2, x1:x2] = 1
    return mask


def model_class_names(model: YOLO) -> list[str]:
    names = model.names
    if isinstance(names, dict):
        return [str(names[idx]) for idx in sorted(names)]
    return [str(name) for name in names]


def resolve_category_mapping(coco_data: dict, model: YOLO) -> dict[int, int]:
    # 保留原始 COCO category id 作為 class index。
    # 這會和使用 --preserve-category-ids 準備出的資料集保持一致。
    # Preserve the original COCO category id as the class index.
    # This matches datasets prepared with --preserve-category-ids.
    sorted_cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    model_names = model_class_names(model)

    idx_to_cat_id: dict[int, int] = {}
    for cat in sorted_cats:
        cat_id = int(cat["id"])
        if cat_id >= len(model_names):
            raise RuntimeError(
                f"Model class index range is too small for category id {cat_id}. "
                f"Model exposes {len(model_names)} classes."
            )

        idx_to_cat_id[cat_id] = cat_id
        model_name = model_names[cat_id].strip()
        cat_name = str(cat["name"]).strip()
        if model_name != cat_name:
            print(
                f"Warning: class name mismatch at category id {cat_id}: "
                f"model='{model_name}' dataset='{cat_name}'. "
                "Using preserved category ids for mapping."
            )
    return idx_to_cat_id


def evaluate_segmentation(args: argparse.Namespace) -> dict[str, float] | None:
    input_root = Path(args.input_root)
    ann_path = input_root / args.split / "_annotations.coco.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    coco_data = load_coco(ann_path)
    images = coco_data.get("images", [])
    if args.max_images is not None:
        images = images[: args.max_images]

    model = YOLO(args.model)
    idx_to_cat_id = resolve_category_mapping(coco_data, model)

    segmenter = SAM2BoxSegmenter(
        model_id=args.sam2_model_id,
        checkpoint_path=args.sam2_checkpoint,
        config_path=args.sam2_config,
        device=args.device,
        multimask_output=not args.single_mask,
    )

    coco_gt = COCO(str(ann_path))
    coco_results: list[dict] = []
    debug_rows: list[dict] = []
    skipped_masks = 0
    fallback_masks = 0
    total_boxes = 0

    split_dir = input_root / args.split

    for image_info in tqdm(images, desc=f"Evaluating segm [{args.split}]"):
        # 先在原圖上執行 YOLO detection，再用同一張原圖做 SAM2 box-prompt segmentation。
        # 這樣產生的 mask 會和 ground-truth segmentation 使用同一組座標系統。
        # Run YOLO detection on the original image, then use the same image for
        # SAM2 box-prompt segmentation so predicted masks stay aligned with GT coordinates.
        image_path = resolve_image_path(split_dir, image_info["file_name"])
        image_rgb = np.array(Image.open(image_path).convert("RGB"))
        height, width = image_rgb.shape[:2]

        segmenter.set_image(image_rgb)
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            imgsz=args.imgsz,
            verbose=False,
        )

        if not results:
            continue

        # 讀出這張圖的 YOLO detection 結果，包括 bbox、confidence 和 class index。
        # Pull the YOLO detection results for this image, including boxes,
        # confidence scores, and class indices.
        boxes = results[0].boxes
        if boxes is None:
            continue

        # Ultralytics 回傳的是 torch tensor，這裡先脫離計算圖、搬到 CPU，再轉成 numpy。
        # 這樣後面的 Python / numpy 處理會比較直接。
        # xyxy: 每個 detection 的 [x1, y1, x2, y2] 邊界框座標
        # conf: 每個 detection 的信心分數
        # cls:  每個 detection 的類別索引
        # Ultralytics returns torch tensors, so detach them from the graph,
        # move them to CPU, and convert to numpy for plain Python/numpy processing.
        # xyxy: [x1, y1, x2, y2] box coordinates for each detection
        # conf: confidence score for each detection
        # cls:  class index for each detection
        xyxy_list = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        conf_list = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.empty((0,))
        cls_list = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.empty((0,))

        for det_idx, (xyxy, score, class_idx) in enumerate(
            zip(xyxy_list, conf_list, cls_list),
            start=1,
        ):
            class_id = int(class_idx)
            if class_id not in idx_to_cat_id:
                continue

            total_boxes += 1

            # 可在送進 SAM2 前先把 bbox 稍微向外擴張，
            # 讓 prompt 有機會覆蓋到框外一點點的物體邊界。
            # Optionally pad the predicted bbox before sending it to SAM2,
            # so the prompt can capture object boundaries slightly outside the box.
            prompt_box = expand_box_xyxy(
                xyxy.astype(np.float32),
                width=width,
                height=height,
                pad_ratio=args.box_pad_ratio,
            )

            mask = None
            try:
                mask = segmenter.predict_mask(prompt_box)
                mask = (mask > 0).astype(np.uint8)
            except Exception as exc:
                print(
                    f"Warning: SAM2 failed on {image_info['file_name']} "
                    f"det {det_idx}: {exc}"
                )

            if mask is None or int(mask.sum()) < args.min_mask_area:
                if args.empty_mask_policy == "skip":
                    skipped_masks += 1
                    continue

                # 如果 SAM2 沒有產生有效 mask，就退回使用矩形 box mask。
                # 這樣仍可把純 detection 的結果投影到 segmentation benchmark 上評估。
                # If SAM2 does not produce a valid mask, fall back to a rectangular
                # box mask so plain detection boxes can still be measured on the
                # segmentation benchmark.
                fallback_masks += 1
                mask = box_xyxy_to_mask(prompt_box, height=height, width=width)

            segmentation, area, bbox_xywh = build_segmentation(
                mask=mask,
                segmentation_format=args.segmentation_format,
                fallback_box=prompt_box,
            )

            coco_results.append(
                {
                    "image_id": int(image_info["id"]),
                    "category_id": idx_to_cat_id[class_id],
                    "segmentation": mask_to_rle(mask),
                    "score": float(score),
                }
            )

            debug_rows.append(
                {
                    "image_id": int(image_info["id"]),
                    "image_file": image_info["file_name"],
                    "category_id": idx_to_cat_id[class_id],
                    "score": float(score),
                    "bbox": [float(v) for v in bbox_xywh],
                    "area": float(area),
                    "segmentation_type": "rle",
                    "raw_segmentation_type": "rle" if args.segmentation_format == "rle" else "polygon",
                    "segmentation": segmentation,
                }
            )

    print(f"\nTotal YOLO boxes: {total_boxes}")
    print(f"Predicted masks:  {len(coco_results)}")
    print(f"Fallback masks:   {fallback_masks}")
    print(f"Skipped masks:    {skipped_masks}")

    if args.pred_json_out:
        pred_json_path = Path(args.pred_json_out)
        pred_json_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_json_path.open("w", encoding="utf-8") as f:
            json.dump(debug_rows, f, ensure_ascii=False, indent=2)
        print(f"Prediction JSON saved to: {pred_json_path}")

    if not coco_results:
        print("No segmentation predictions were produced.")
        return None

    # 只評估這次實際處理到的 image 子集合。
    # 當 --max-images 用於 smoke test 或除錯時，這點尤其重要。
    # Evaluate only the subset of images processed in this run.
    # This is especially important when --max-images is used for smoke tests or debugging.
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.imgIds = [int(image["id"]) for image in images]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        "mAP50_95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "AR@100": float(coco_eval.stats[8]),
    }
    print(
        f"\nSegmentation metrics on {args.split}: "
        f"mAP50={metrics['mAP50']:.4f}, "
        f"mAP50-95={metrics['mAP50_95']:.4f}, "
        f"AR@100={metrics['AR@100']:.4f}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv11 detection predictions as segmentation via SAM2"
    )
    parser.add_argument("--model", required=True, help="Path to YOLOv11 detection weights")
    parser.add_argument("--input-root", required=True, help="COCO segmentation dataset root")
    parser.add_argument("--split", default="test", help="Dataset split: train/valid/test")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image")
    parser.add_argument("--imgsz", type=int, required=True, help="Resize image size used during YOLO inference")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--sam2-checkpoint", default=None, help="Optional local SAM2 checkpoint")
    parser.add_argument("--sam2-config", default=None, help="Optional local SAM2 config")
    parser.add_argument("--box-pad-ratio", type=float, default=0.0, help="Expand predicted bbox before SAM2 prompt")
    parser.add_argument("--segmentation-format", choices=["polygon", "rle"], default="polygon")
    parser.add_argument("--empty-mask-policy", choices=["box", "skip"], default="box")
    parser.add_argument("--min-mask-area", type=int, default=16, help="Minimum valid predicted mask area")
    parser.add_argument("--max-images", type=int, default=None, help="Useful for smoke tests")
    parser.add_argument("--single-mask", action="store_true", help="Disable SAM2 multimask output")
    parser.add_argument("--pred-json-out", default=None, help="Optional path to save per-instance prediction JSON")
    args = parser.parse_args()

    try:
        import sam2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The sam2 package is required. Install it with:\n"
            "pip install git+https://github.com/facebookresearch/sam2.git"
        ) from exc

    evaluate_segmentation(args)


if __name__ == "__main__":
    main()
