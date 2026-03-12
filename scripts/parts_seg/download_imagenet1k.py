"""
Download ImageNet-1K (ILSVRC2012) via Kaggle for use with PartImageNet++.

Prerequisites:
  1. Create a Kaggle account: https://www.kaggle.com
  2. Join the ImageNet Object Localization Challenge (required to accept terms):
     https://www.kaggle.com/competitions/imagenet-object-localization-challenge
  3. Get API credentials: Kaggle account → Settings → Create New Token.
     Place kaggle.json at ~/.kaggle/kaggle.json (or set KAGGLE_CONFIG_DIR).

Usage (from repo root; run in terminal to see progress):
  python -m scripts.parts_seg.download_imagenet1k
  python -m scripts.parts_seg.download_imagenet1k --output_dir data/ImageNet-1K

After download, set PartImageNet++ image_root to the directory that contains
the n* class folders (e.g. output_dir/train if the archive extracts that way).
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download ImageNet-1K (ILSVRC2012) via Kaggle into the data folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the dataset (default: data/ImageNet-1K)",
    )
    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Only download the zip; do not extract",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "data" / "ImageNet-1K"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
    except ImportError:
        print("Kaggle API is required. Install with: pip install kaggle", file=sys.stderr)
        sys.exit(1)

    print("Downloading ImageNet Object Localization Challenge from Kaggle.")
    print("This is a large download (~150GB+). Do not pipe the output to see progress.")
    print(f"Output directory: {output_dir}")

    # Download (no pipe so user sees progress)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        "imagenet-object-localization-challenge",
        "-p",
        str(output_dir),
    ]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(
            "Download failed. Ensure you have joined the competition and set ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        sys.exit(ret.returncode)

    if args.skip_extract:
        print("Skipping extraction (--skip_extract). Extract the zip manually.")
        return

    # Find the downloaded zip and extract
    zips = list(output_dir.glob("*.zip"))
    if not zips:
        print("No zip found in output dir. You may need to extract manually.")
        return
    # Often a single zip like imagenet-object-localization-challenge.zip
    zip_path = zips[0]
    print(f"Extracting {zip_path} ...")
    subprocess.run(
        ["unzip", "-o", str(zip_path), "-d", str(output_dir)],
        check=False,
    )
    print("Done.")
    print("For PartImageNet++, set image_root to the folder that contains the n* class dirs")
    print(f"  (e.g. {output_dir}/train or {output_dir}, depending on archive layout).")


if __name__ == "__main__":
    main()
