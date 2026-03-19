"""
Download a Roboflow object detection dataset in COCO format.

This is the shared implementation used by other pipelines in this repo.

Example:
    python -m scripts.tools.download_roboflow_coco \
      --dataset-url https://universe.roboflow.com/myworkspacename/hiod-1fqtj \
      --version 1 \
      --api-key <ROBOFLOW_API_KEY> \
      --output-dir data/hiod_coco
"""
import argparse
import json
import os
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_API_URL = "https://api.roboflow.com"
CREDENTIALS_FILE = "roboflow_credentials.json"


def _clean_dict(values: dict) -> dict:
    return {key: value for key, value in values.items() if value not in (None, "")}


def parse_universe_url(dataset_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(dataset_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Could not parse workspace/project from dataset URL: {dataset_url}"
        )
    return parts[0], parts[1]


def load_credentials(credentials_file: str | None = None, cli_overrides: dict | None = None) -> dict:
    cli_overrides = _clean_dict(cli_overrides or {})

    env_creds = _clean_dict(
        {
            "ROBOFLOW_API_KEY": os.getenv("ROBOFLOW_API_KEY"),
            "ROBOFLOW_API_URL": os.getenv("ROBOFLOW_API_URL"),
        }
    )

    candidate_paths = []
    if credentials_file:
        candidate_paths.append(Path(credentials_file))
    env_credentials_path = os.getenv("ROBOFLOW_CREDENTIALS_FILE")
    if env_credentials_path:
        candidate_paths.append(Path(env_credentials_path))
    candidate_paths.append(Path(CREDENTIALS_FILE))

    file_creds = {}
    for candidate in candidate_paths:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as f:
                file_creds = json.load(f)
            break

    creds = {}
    creds.update(file_creds)
    creds.update(env_creds)
    creds.update(cli_overrides)

    if "ROBOFLOW_API_KEY" not in creds:
        raise FileNotFoundError(
            "Roboflow API key not found. Provide --api-key, --credentials, or set "
            "ROBOFLOW_API_KEY / ROBOFLOW_CREDENTIALS_FILE."
        )

    return creds


def get_download_url(
    workspace: str,
    project: str,
    version: str | int,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    format_name: str = "coco",
) -> str:
    workspace = workspace.strip("/")
    project = project.strip("/")
    return f"{api_url.rstrip('/')}/{workspace}/{project}/{version}/{format_name}?api_key={api_key}"


def download_dataset(
    dataset_url: str,
    version: str | int,
    output_dir: str,
    credentials_file: str | None = None,
    api_key: str | None = None,
    api_url: str = DEFAULT_API_URL,
    format_name: str = "coco",
) -> Path:
    workspace, project = parse_universe_url(dataset_url)
    creds = load_credentials(
        credentials_file=credentials_file,
        cli_overrides={
            "ROBOFLOW_API_KEY": api_key,
            "ROBOFLOW_API_URL": api_url,
        },
    )
    download_url = get_download_url(
        workspace=workspace,
        project=project,
        version=version,
        api_key=creds["ROBOFLOW_API_KEY"],
        api_url=creds.get("ROBOFLOW_API_URL", DEFAULT_API_URL),
        format_name=format_name,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / "dataset.zip"

    print(f"Preparing Roboflow export for {workspace}/{project} v{version}...")
    print(f"API URL: {download_url}")

    meta = None
    for attempt in range(20):
        with urllib.request.urlopen(download_url) as resp:
            meta = json.loads(resp.read().decode("utf-8"))

        if "export" in meta:
            break

        progress = meta.get("progress", 0)
        print(f"  Export generating... ({progress}%) [attempt {attempt + 1}/20]")
        import time
        time.sleep(5)
    else:
        raise RuntimeError(
            f"Roboflow export never became ready. Last response:\n{json.dumps(meta, indent=2)}"
        )

    export_link = meta["export"]["link"]
    print(f"Downloading archive from: {export_link}")

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100.0 / total_size, 100.0)
            print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(export_link, zip_path, reporthook=reporthook)
    print()

    print(f"Extracting to: {output_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_path)

    zip_path.unlink(missing_ok=True)

    print("\nExtracted splits:")
    for ann_path in sorted(output_path.rglob("_annotations.coco.json")):
        split_dir = ann_path.parent
        image_count = len(
            list(split_dir.glob("*.jpg"))
            + list(split_dir.glob("*.jpeg"))
            + list(split_dir.glob("*.png"))
        )
        print(f"  {split_dir.name}: {image_count} images [{ann_path}]")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Roboflow COCO detection dataset")
    parser.add_argument("--dataset-url", required=True, help="Roboflow Universe URL")
    parser.add_argument("--version", required=True, help="Roboflow dataset version number")
    parser.add_argument("--output-dir", default="data/roboflow_coco")
    parser.add_argument("--credentials", default=None, help="Path to roboflow_credentials.json")
    parser.add_argument("--api-key", default=None, help="Roboflow API key override")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--format", default="coco", help="Roboflow export format, default: coco")
    args = parser.parse_args()

    download_dataset(
        dataset_url=args.dataset_url,
        version=args.version,
        output_dir=args.output_dir,
        credentials_file=args.credentials,
        api_key=args.api_key,
        api_url=args.api_url,
        format_name=args.format,
    )


if __name__ == "__main__":
    main()
