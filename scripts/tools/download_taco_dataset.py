"""
下載 TACO 的 Flickr 圖片，並在寫入時直接進行 auto-orient。
Download TACO images from Flickr and auto-orient them during save.

使用方式 Usage:
    python -m scripts.tools.download_taco_dataset

預設流程 Default workflow:
1. 下載釋出的 taco_dataset zip 壓縮檔
2. 解壓縮到資料集目錄
3. 自動辨識實際的資料集根目錄
4. 尋找 train/valid/test 的 annotation 檔案
5. 為各 split 平行下載缺少的圖片
6. 呼叫 scripts.tools.auto_orient_tool 套用 auto orientation
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from scripts.tools.auto_orient_tool import auto_orient_and_strip

DEFAULT_ARCHIVE_URL = (
    "https://github.com/chenp6/SegmentationTask/releases/download/add_taco_dataset/"
    "taco_dataset.zip"
)
DEFAULT_ARCHIVE_NAME = "data/taco_dataset.zip"
DEFAULT_DATASET_DIR = "data/taco_dataset"
DEFAULT_NUM_WORKERS = 8
SPLITS = ("train", "valid", "test")
ANNOTATION_FILENAMES = ("_annotation.coco.json", "_annotations.coco.json")


def parse_args() -> argparse.Namespace:
    """解析命令列參數。 Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "下載 TACO 的 Flickr 圖片並在儲存時自動校正方向。 "
            "Download TACO images from Flickr and auto-orient them during save."
        )
    )
    parser.add_argument(
        "--archive-url",
        default=DEFAULT_ARCHIVE_URL,
        help="壓縮檔下載網址。 Archive download URL.",
    )
    parser.add_argument(
        "--archive-path",
        default=DEFAULT_ARCHIVE_NAME,
        help="壓縮檔儲存路徑。 Local path used to store the downloaded archive.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="解壓縮後的資料集目錄。 Directory used to extract the dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="每次 HTTP 請求的逾時秒數。 HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="平行下載的執行緒數量。 Number of worker threads for parallel downloads.",
    )
    return parser.parse_args()


def download_file(url: str, destination_path: Path, timeout: float) -> None:
    """下載檔案到指定路徑。 Download a file to the destination path."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    """解壓縮 zip 壓縮檔到指定資料夾。 Extract a zip archive into the target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)


def prepare_default_dataset(
    archive_url: str,
    archive_path: Path,
    dataset_dir: Path,
    timeout: float,
) -> None:
    """準備預設資料集，必要時自動下載與解壓。 Prepare the default dataset by downloading and extracting when needed."""
    if not archive_path.exists():
        print(f"Downloading archive: {archive_url}")
        download_file(archive_url, archive_path, timeout)

    if not dataset_dir.exists():
        print(f"Extracting archive to: {dataset_dir}")
        extract_archive(archive_path, dataset_dir)


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    """
    自動辨識實際資料集根目錄。
    Resolve the actual dataset root directory automatically.

    支援兩種常見解壓結果：
    1. data/taco_dataset/train
    2. data/taco_dataset/taco_dataset/train
    """
    direct_split_dirs = [dataset_dir / split for split in SPLITS]
    if all(path.exists() for path in direct_split_dirs):
        return dataset_dir

    nested_dir = dataset_dir / dataset_dir.name
    nested_split_dirs = [nested_dir / split for split in SPLITS]
    if nested_dir.exists() and all(path.exists() for path in nested_split_dirs):
        return nested_dir

    return dataset_dir


def find_split_annotation_paths(dataset_dir: Path) -> list[Path]:
    """尋找 train/valid/test 的 annotation 檔案。 Find annotation files for train/valid/test."""
    annotation_paths: list[Path] = []
    for split in SPLITS:
        split_dir = dataset_dir / split
        annotation_path = None

        for filename in ANNOTATION_FILENAMES:
            candidate = split_dir / filename
            if candidate.exists():
                annotation_path = candidate
                break

        if annotation_path is None:
            raise FileNotFoundError(
                f"Annotation file not found for split '{split}' under {split_dir}"
            )
        annotation_paths.append(annotation_path)

    return annotation_paths


def download_and_orient_image(
    image_url: str,
    destination_path: Path,
    timeout: float,
) -> None:
    """下載單張圖片並套用 auto-orient。 Download one image and apply auto orientation."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(image_url, timeout=timeout)
    response.raise_for_status()

    # 保留原始副檔名，讓 PIL 能正確推斷暫存檔格式。
    # Keep the original image suffix so PIL can infer the temporary file format.
    suffix = destination_path.suffix or ".jpg"
    temp_path = destination_path.with_name(
        f"{destination_path.stem}.download_tmp{suffix}"
    )
    try:
        img = Image.open(BytesIO(response.content))
        img.save(temp_path)
        auto_orient_and_strip(temp_path, destination_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def print_progress(prefix: str, current_count: int, total_count: int) -> None:
    """輸出簡單進度條。 Print a simple progress bar."""
    bar_size = 30
    filled = int(bar_size * current_count / total_count) if total_count else 0
    sys.stdout.write(
        "{}[{}{}] - {}/{}\r".format(
            prefix,
            "=" * filled,
            "." * (bar_size - filled),
            current_count,
            total_count,
        )
    )
    sys.stdout.flush()


def process_single_image(image: dict, dataset_dir: Path, timeout: float) -> None:
    """處理單張圖片下載任務。 Process a single image download task."""
    file_name = image["file_name"]
    image_url = image.get("flickr_url") or image.get("flickr_640_url")
    destination_path = dataset_dir / file_name

    if destination_path.exists():
        return

    if not image_url:
        raise ValueError(f"Missing Flickr URL for image: {file_name}")

    download_and_orient_image(
        image_url=image_url,
        destination_path=destination_path,
        timeout=timeout,
    )


def process_annotation_file(dataset_path: Path, timeout: float, num_workers: int) -> None:
    """處理單一 annotation 檔案，平行下載缺少的圖片。 Process one annotation file and download missing images in parallel."""
    dataset_dir = dataset_path.parent

    print(f"Processing annotation file: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    images = annotations.get("images", [])
    total_count = len(images)
    completed_count = 0

    max_workers = max(1, min(num_workers, total_count if total_count else 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(process_single_image, image, dataset_dir, timeout): image
            for image in images
        }

        for future in as_completed(future_to_image):
            image = future_to_image[future]
            future.result()
            completed_count += 1
            print_progress(
                prefix=f"{dataset_dir.name}: ",
                current_count=completed_count,
                total_count=total_count,
            )

    sys.stdout.write(f"{dataset_dir.name}: Finished\n")


def main() -> None:
    """主程式入口。 Program entry point."""
    args = parse_args()
    archive_path = Path(args.archive_path)
    dataset_dir = Path(args.dataset_dir)

    prepare_default_dataset(
        archive_url=args.archive_url,
        archive_path=archive_path,
        dataset_dir=dataset_dir,
        timeout=args.timeout,
    )
    dataset_dir = resolve_dataset_dir(dataset_dir)
    annotation_paths = find_split_annotation_paths(dataset_dir)

    print(
        "Note. If the connection is interrupted, running this command again "
        "will continue from the remaining files."
    )

    for annotation_path in annotation_paths:
        process_annotation_file(
            annotation_path,
            timeout=args.timeout,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
