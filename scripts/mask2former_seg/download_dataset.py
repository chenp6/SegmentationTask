"""
Download hospital segmentation dataset from Roboflow using API v1 in COCO format.

Usage:
    python -m scripts.mask2former_seg.download_dataset
"""
import json
import os
import urllib.request
import zipfile
from pathlib import Path

CREDENTIALS_FILE = "roboflow_credentials.json"


def load_credentials() -> dict:
    with open(CREDENTIALS_FILE) as f:
        return json.load(f)


def get_download_url(creds: dict) -> str:
    """
    Build Roboflow API v1 download URL for COCO format.
    URL format: https://api.roboflow.com/{workspace}/{project}/{version}/coco?api_key={key}

    Note: In the credentials file, WORKSPACE=hospital-90c5d-jmw1o (the project slug)
    and PROJECT=yolotest-6mtuj (the workspace slug) - these match the Roboflow URL pattern
    where workspace comes first.
    """
    api_key = creds["ROBOFLOW_API_KEY"]
    # Roboflow URL pattern: /{workspace}/{project}/{version}/coco
    # From user: "yolotest-6mtuj/hospital-90c5d-jmw1o" -> workspace/project
    workspace = creds.get("ROBOFLOW_PROJECT", "yolotest-6mtuj")   # workspace slug
    project   = creds.get("ROBOFLOW_WORKSPACE", "hospital-90c5d-jmw1o")  # project slug
    version   = creds.get("ROBOFLOW_VERSION", "1")
    base_url  = creds.get("ROBOFLOW_API_URL", "https://api.roboflow.com")

    url = f"{base_url}/{workspace}/{project}/{version}/coco?api_key={api_key}"
    return url


def download_dataset(output_dir: str = "data/hospital_coco") -> str:
    """
    Download the dataset from Roboflow and unzip to output_dir.
    Handles the async export flow: keeps polling until the link is ready.
    Returns the path to the extracted data directory.
    """
    import time
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    creds = load_credentials()
    url = get_download_url(creds)

    print(f"Fetching Roboflow dataset link...")
    print(f"  URL: {url[:80]}...")

    # Roboflow may return {"progress": 0} the first time, triggering export generation.
    # Poll until we get {"export": {"link": "..."}}
    max_retries = 20
    for attempt in range(max_retries):
        with urllib.request.urlopen(url) as resp:
            meta = json.loads(resp.read().decode())

        if "export" in meta:
            break   # Export is ready

        progress = meta.get("progress", 0)
        print(f"  Export generating... ({progress}%)  [attempt {attempt+1}/{max_retries}]")
        time.sleep(5)
    else:
        print(f"API response: {json.dumps(meta, indent=2)}")
        raise RuntimeError(
            "Roboflow export did not become ready after polling. "
            "Check workspace/project/version in credentials."
        )

    export_link = meta["export"]["link"]
    print(f"Downloading dataset from: {export_link[:80]}...")

    # Download the zip file
    zip_path = os.path.join(output_dir, "dataset.zip")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded * 100 / total_size, 100)
            print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(export_link, zip_path, reporthook=reporthook)
    print()

    # Extract
    print(f"Extracting to {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    os.remove(zip_path)

    # Show what was extracted
    print("\nExtracted files:")
    for p in sorted(Path(output_dir).rglob("_annotations.coco.json")):
        count = len(list(p.parent.glob("*.jpg")) + list(p.parent.glob("*.png")) + list(p.parent.glob("*.jpeg")))
        print(f"  {p.parent.name:10s}: {count} images  [{p}]")

    print(f"\nDataset ready at: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/hospital_coco")
    args = parser.parse_args()
    download_dataset(args.output_dir)