"""
Generate COCO keypoints from polygon segmentation boundaries.

This tool reads a COCO annotation json, samples points along each polygon boundary,
and writes keypoints back into COCO format:
- annotations[].keypoints (x, y, v triplets)
- annotations[].num_keypoints
- categories[].keypoints (labels)

Notes:
- Only polygon segmentations (list format) are converted.
- RLE segmentation annotations are skipped.

Example:
  python -m scripts.tools.coco_polygon_to_keypoints \
    --input-json data/myset/train/_annotations.coco.json \
    --output-json data/myset/train/_annotations.kpt.coco.json \
    --num-keypoints 8 \
    --keypoint-labels top,left,bottom,right,tl,tr,bl,br
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def segmentation_to_polygons(segmentation: list) -> List[List[Tuple[float, float]]]:
    polygons: List[List[Tuple[float, float]]] = []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
            continue
        pts = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
        polygons.append(pts)
    return polygons


def perimeter_lengths(points: Sequence[Tuple[float, float]]) -> tuple[list[float], float]:
    if len(points) < 2:
        return [], 0.0
    lengths: list[float] = []
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        lengths.append(seg_len)
        total += seg_len
    return lengths, total


def sample_polygon_equally(points: Sequence[Tuple[float, float]], num_points: int) -> list[Tuple[float, float]]:
    if len(points) < 3 or num_points <= 0:
        return []

    seg_lengths, perimeter = perimeter_lengths(points)
    if perimeter <= 0:
        return []

    targets = [perimeter * i / num_points for i in range(num_points)]
    sampled: list[Tuple[float, float]] = []

    cum = 0.0
    seg_idx = 0
    for t in targets:
        while seg_idx < len(seg_lengths) and cum + seg_lengths[seg_idx] < t:
            cum += seg_lengths[seg_idx]
            seg_idx += 1

        if seg_idx >= len(seg_lengths):
            sampled.append(points[-1])
            continue

        x1, y1 = points[seg_idx]
        x2, y2 = points[(seg_idx + 1) % len(points)]
        seg_len = seg_lengths[seg_idx]
        if seg_len == 0:
            sampled.append((x1, y1))
            continue

        local_t = (t - cum) / seg_len
        x = x1 + (x2 - x1) * local_t
        y = y1 + (y2 - y1) * local_t
        sampled.append((x, y))

    return sampled


def format_coco_keypoints(points: Iterable[Tuple[float, float]], visibility: int) -> list[float]:
    flat: list[float] = []
    v = int(max(0, min(2, visibility)))
    for x, y in points:
        flat.extend([round(float(x), 2), round(float(y), 2), v])
    return flat


def parse_keypoint_labels(raw: str | None, num_keypoints: int, prefix: str) -> list[str]:
    if raw:
        labels = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        labels = []

    if not labels:
        labels = [f"{prefix}{i+1}" for i in range(num_keypoints)]

    if len(labels) != num_keypoints:
        raise ValueError(
            f"keypoint labels count ({len(labels)}) must equal --num-keypoints ({num_keypoints})"
        )
    return labels


def convert(
    coco: dict,
    num_keypoints: int,
    labels: list[str],
    visibility: int,
    overwrite_existing: bool,
) -> tuple[int, int, int]:
    converted = 0
    skipped_non_polygon = 0
    skipped_invalid = 0

    for ann in coco.get("annotations", []):
        if (not overwrite_existing) and isinstance(ann.get("keypoints"), list):
            continue

        seg = ann.get("segmentation")
        if not isinstance(seg, list):
            skipped_non_polygon += 1
            continue

        polygons = segmentation_to_polygons(seg)
        if not polygons:
            skipped_invalid += 1
            continue

        # Use the largest polygon when there are multiple parts.
        best_poly = max(polygons, key=polygon_area)
        points = sample_polygon_equally(best_poly, num_keypoints)
        if len(points) != num_keypoints:
            skipped_invalid += 1
            continue

        ann["keypoints"] = format_coco_keypoints(points, visibility)
        ann["num_keypoints"] = num_keypoints
        converted += 1

    for cat in coco.get("categories", []):
        cat["keypoints"] = list(labels)
        if "skeleton" not in cat or not isinstance(cat["skeleton"], list):
            cat["skeleton"] = []

    return converted, skipped_non_polygon, skipped_invalid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate COCO keypoints from polygon boundaries")
    parser.add_argument("--input-json", required=True, help="Input COCO annotation json")
    parser.add_argument("--output-json", required=True, help="Output COCO annotation json")
    parser.add_argument("--num-keypoints", type=int, required=True, help="Number of keypoints per instance")
    parser.add_argument(
        "--keypoint-labels",
        default=None,
        help="Comma-separated keypoint labels. If omitted, auto-generate with --keypoint-prefix.",
    )
    parser.add_argument(
        "--keypoint-prefix",
        default="kp",
        help="Prefix for auto-generated labels when --keypoint-labels is omitted (default: kp)",
    )
    parser.add_argument(
        "--visibility",
        type=int,
        default=2,
        help="COCO visibility value for generated points: 0/1/2 (default: 2)",
    )
    parser.add_argument(
        "--keep-existing-keypoints",
        action="store_true",
        help="Do not overwrite annotations that already contain keypoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_keypoints <= 0:
        raise ValueError("--num-keypoints must be > 0")

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)

    coco = load_json(input_json)
    labels = parse_keypoint_labels(args.keypoint_labels, args.num_keypoints, args.keypoint_prefix)

    converted, skipped_non_polygon, skipped_invalid = convert(
        coco=coco,
        num_keypoints=args.num_keypoints,
        labels=labels,
        visibility=args.visibility,
        overwrite_existing=not args.keep_existing_keypoints,
    )

    save_json(output_json, coco)

    print(f"done -> {output_json}")
    print(f"  converted annotations:     {converted}")
    print(f"  skipped non-polygon anns:  {skipped_non_polygon}")
    print(f"  skipped invalid polygons:  {skipped_invalid}")
    print(f"  keypoint labels:           {labels}")


if __name__ == "__main__":
    main()
