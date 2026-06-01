"""
Upload a local COCO-format dataset to Roboflow (images + annotations).

Expected layout:
  <dataset_dir>/
    train/_annotations.coco.json
    valid/_annotations.coco.json
    test/_annotations.coco.json   # optional

Usage:
  python -m scripts.tools.upload_roboflow_coco \
    --dataset-dir data/medbin_dataset \
    --credentials roboflow_credentials.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from roboflow import Roboflow


DEFAULT_CREDENTIALS_FILE = "roboflow_credentials.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"COCO annotation must be a JSON object: {path}")
    return data


def _ensure_coco_metadata(ann_path: Path, autofix: bool) -> None:
    data = _load_json(ann_path)
    required = ("images", "annotations", "categories")
    missing_required = [k for k in required if k not in data]
    if missing_required:
        missing = ", ".join(missing_required)
        raise RuntimeError(f"Invalid COCO JSON (missing {missing}): {ann_path}")

    changed = False
    if "info" not in data:
        if not autofix:
            raise RuntimeError(
                "Invalid COCO JSON for Roboflow: missing top-level 'info' in "
                f"{ann_path}. Re-export annotations or run with --autofix-coco-metadata."
            )
        data["info"] = {
            "description": "autofixed for Roboflow COCO upload",
            "version": "1.0",
            "year": 2026,
            "contributor": "",
            "date_created": "2026-05-10",
        }
        changed = True

    if "licenses" not in data:
        if not autofix:
            raise RuntimeError(
                "Invalid COCO JSON for Roboflow: missing top-level 'licenses' in "
                f"{ann_path}. Re-export annotations or run with --autofix-coco-metadata."
            )
        data["licenses"] = []
        changed = True

    if changed:
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[fix] Added missing COCO metadata: {ann_path}")


def load_credentials(credentials_path: str) -> dict:
    with open(credentials_path, "r", encoding="utf-8") as f:
        creds = json.load(f)
    return creds


def _resolve_config(args: argparse.Namespace) -> tuple[str, str, str]:
    creds_path = args.credentials or DEFAULT_CREDENTIALS_FILE
    creds = load_credentials(creds_path)

    api_key = args.api_key or creds.get("ROBOFLOW_API_KEY")
    workspace = args.workspace or creds.get("ROBOFLOW_WORKSPACE")
    project = args.project or creds.get("ROBOFLOW_PROJECT")

    missing = []
    if not api_key:
        missing.append("ROBOFLOW_API_KEY / --api-key")
    if not workspace:
        missing.append("ROBOFLOW_WORKSPACE / --workspace")
    if not project:
        missing.append("ROBOFLOW_PROJECT / --project")
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required config: {missing_str}")

    return api_key, workspace, project


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _validate_split(dataset_dir: Path, split: str, required: bool, autofix_coco_metadata: bool) -> None:
    split_dir = dataset_dir / split
    ann_path = split_dir / "_annotations.coco.json"

    if not split_dir.exists():
        if required:
            raise FileNotFoundError(f"Missing required split directory: {split_dir}")
        return

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {ann_path}")

    _ensure_coco_metadata(ann_path, autofix=autofix_coco_metadata)

    image_count = _count_images(split_dir)
    if image_count == 0:
        raise RuntimeError(f"No images found in split directory: {split_dir}")

    print(f"[ok] {split:5s}: {image_count:5d} images, annotation={ann_path}")


def validate_coco_layout(dataset_dir: str, required_splits: Iterable[str], autofix_coco_metadata: bool) -> Path:
    root = Path(dataset_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    required_splits_set = set(required_splits)
    for split in ("train", "valid", "test"):
        _validate_split(
            root,
            split,
            required=split in required_splits_set,
            autofix_coco_metadata=autofix_coco_metadata,
        )

    return root


def upload_coco_dataset(
    dataset_dir: str,
    api_key: str,
    workspace: str,
    project: str,
    num_workers: int = 8,
    project_type: str = "instance-segmentation",
    autofix_coco_metadata: bool = False,
) -> None:
    root = validate_coco_layout(
        dataset_dir,
        required_splits=("train", "valid"),
        autofix_coco_metadata=autofix_coco_metadata,
    )

    print("\nConnecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)

    print("Uploading dataset...")
    print(f"  dataset_dir : {root}")
    print(f"  workspace   : {workspace}")
    print(f"  project     : {project}")
    print(f"  project_type: {project_type}")
    print(f"  workers     : {num_workers}")

    ws.upload_dataset(
        str(root),
        project,
        num_workers=num_workers,
        project_type=project_type,
    )
    print("\nUpload completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload COCO dataset to Roboflow.")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset root with train/valid/test splits")
    parser.add_argument("--credentials", default=None, help="Path to roboflow_credentials.json")
    parser.add_argument("--api-key", default=None, help="Roboflow API key (overrides credentials file)")
    parser.add_argument("--workspace", default=None, help="Roboflow workspace slug (overrides credentials file)")
    parser.add_argument("--project", default=None, help="Roboflow project slug (overrides credentials file)")
    parser.add_argument(
        "--project-type",
        default="instance-segmentation",
        choices=("object-detection", "instance-segmentation", "classification"),
        help="Target Roboflow project type",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of upload workers")
    parser.add_argument(
        "--autofix-coco-metadata",
        action="store_true",
        help="Auto-add missing top-level COCO 'info'/'licenses' fields in _annotations.coco.json files",
    )
    args = parser.parse_args()

    # Allow environment variable override if present.
    args.api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")

    api_key, workspace, project = _resolve_config(args)
    upload_coco_dataset(
        dataset_dir=args.dataset_dir,
        api_key=api_key,
        workspace=workspace,
        project=project,
        num_workers=args.workers,
        project_type=args.project_type,
        autofix_coco_metadata=args.autofix_coco_metadata,
    )


if __name__ == "__main__":
    main()
