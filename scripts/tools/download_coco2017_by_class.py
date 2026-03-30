import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

SPLIT_EXPORT_NAMES = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 samples with FiftyOne and export them in COCO folder format"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where the exported COCO-format dataset should be written",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "bowl",
            "laptop",
            "cell phone",
            "remote",
            "book",
            "backpack",
            "handbag",
            "suitcase",
            "toothbrush",
        ],
        help='Class names to filter, for example: --classes bowl laptop "cell phone" remote',
    )
    return parser.parse_args()


def export_split(dataset: fo.Dataset, output_root: Path, split_name: str) -> None:
    export_split_name = SPLIT_EXPORT_NAMES[split_name]
    split_dir = output_root / export_split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset.export(
        export_dir=str(split_dir),
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(split_dir),
        labels_path=str(split_dir / "_annotations.coco.json"),
        export_media="copy",
    )
    print(f"Exported split: {split_name} -> {split_dir}")


def download_split(split_name: str, cache_root: Path, classes: list[str]) -> fo.Dataset:
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split_name,
        shuffle=(split_name == "train"),
        classes=classes,
        label_types=["detections", "segmentations"],
        dataset_dir=str(cache_root),
    )
    print(f"Downloaded split: {split_name}")
    return dataset


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "_fiftyone_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "validation", "test"):
        dataset = download_split(split_name, cache_root, args.classes)
        export_split(dataset, output_root, split_name)


if __name__ == "__main__":
    main()
