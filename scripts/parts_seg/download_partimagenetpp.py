"""
Download PartImageNet++ annotations from Hugging Face into data/PartImageNetPP.

Run from repo root:
  python -m scripts.parts_seg.download_partimagenetpp
  python -m scripts.parts_seg.download_partimagenetpp --data_dir /path/to/data/PartImageNetPP
"""
import argparse
from pathlib import Path

from .dataset_partimagenetpp import _DEFAULT_DATA_DIR, _ensure_data_dir


def main():
    parser = argparse.ArgumentParser(description="Download PartImageNet++ annotations to data folder")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=f"Target directory (default: {_DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else _DEFAULT_DATA_DIR
    print(f"Downloading PartImageNet++ annotations to {data_dir} ...")
    _ensure_data_dir(data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
