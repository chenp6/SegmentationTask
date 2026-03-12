"""
Prepare a YOLO-seg dataset view from the shared COCO dataset.

Usage:
    python -m scripts.yolov11_seg.prepare_dataset
"""
import argparse
from pathlib import Path

from .config import DataConfig
from .dataset import build_yolo_dataset_from_coco


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO labels and data.yaml from the shared COCO dataset"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="COCO dataset root, e.g. data/hospital_coco",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory for YOLO images/labels/data.yaml. Defaults to <data-root>/yolo",
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default="data.yaml",
        help="Filename for the generated YAML manifest",
    )
    args = parser.parse_args()

    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root

    output_root = args.output_root or str(Path(data_cfg.data_root) / "yolo")
    yaml_path = build_yolo_dataset_from_coco(
        coco_root=data_cfg.data_root,
        output_root=output_root,
        output_yaml=args.output_yaml,
    )

    print(f"Prepared dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
