# DOWNLOAD_DATASET README

對於非 Roboflow 格式的資料集，原則上會先沿用該資料集原本的影像下載來源與基本目錄設計，再整理成統一的 Roboflow 風格資料架構：

```text
train/_annotations.coco.json
valid/_annotations.coco.json
test/_annotations.coco.json
```

在轉換過程中，會依照影像檔案與 `_annotations.coco.json` 之間的相對路徑，調整 `images` 欄位中的 `file_name`，讓標註檔可以正確對應到實際影像位置。

這樣可以把不同來源的資料集快速統一成相同的資料結構，方便後續直接套用各個 training、evaluation 與 conversion pipelines。

For datasets that are not originally in Roboflow format, we generally keep the dataset's original image download source and basic directory arrangement, and then convert the dataset into a unified Roboflow-style layout:

```text
train/_annotations.coco.json
valid/_annotations.coco.json
test/_annotations.coco.json
```

During conversion, the `file_name` field in `images` is rewritten according to the relative path between each image file and `_annotations.coco.json`, so that the annotations correctly point to the actual image locations.

This makes it easier to normalize datasets from different sources into one consistent structure, allowing the downstream training, evaluation, and conversion pipelines to run with minimal extra adjustment.

## Table of contents

- [Hiod dataset preparation](#hiod-dataset-preparation)
- [Medbin dataset preparation](#medbin-dataset-preparation)
- [TACO dataset preparation](#taco-dataset-preparation)
- [COCO dataset preparation](#coco-dataset-preparation)
- [Ward dataset preparation](#ward-dataset-preparation)

## Hiod dataset preparation

```bash
python -m scripts.tools.download_roboflow_coco \
  --dataset-url https://universe.roboflow.com/s-workspace-hmqmj/hiod-rs9ah \
  --version 2 \
  --api-key <API_KEY> \
  --output-dir data/hiod_dataset
```

## Medbin dataset preparation

```bash
python -m scripts.tools.download_roboflow_coco \
  --dataset-url https://universe.roboflow.com/s-workspace-hmqmj/medbin_dataset-azbmq \
  --version 4 \
  --api-key Vffs4D4wHfYxdZ5XAVy4 \
  --output-dir data/medbin_dataset
```

## TACO dataset preparation

`python -m scripts.tools.download_taco_dataset` no longer downloads or unzips the base TACO archive for you.
Prepare the dataset first, then run the script to download missing Flickr images and auto-orient them.

Expected location after extraction:

```text
data/taco_dataset/
  train/
  valid/
  test/
```

Some archives extract into a nested folder like `data/taco_dataset/taco_dataset/`.
That layout is also supported by the script.

### Linux

```bash
mkdir -p data
wget -O data/taco_dataset.zip \
  https://github.com/chenp6/SegmentationTask/releases/download/add_taco_dataset/taco_dataset.zip
unzip data/taco_dataset.zip -d data/
python -m scripts.tools.download_taco_dataset
```

If `wget` is not installed, you can use `curl` instead:

```bash
mkdir -p data
curl -L \
  https://github.com/chenp6/SegmentationTask/releases/download/add_taco_dataset/taco_dataset.zip \
  -o data/taco_dataset.zip
unzip data/taco_dataset.zip -d data/
python -m scripts.tools.download_taco_dataset
```

### PowerShell

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/chenp6/SegmentationTask/releases/download/add_taco_dataset/taco_dataset.zip" `
  -OutFile "data/taco_dataset.zip"
Expand-Archive -Path "data/taco_dataset.zip" -DestinationPath "data" -Force
python -m scripts.tools.download_taco_dataset
```

## COCO dataset preparation

Install FiftyOne first:

```bash
pip install fiftyone
```

Use `scripts.tools.download_coco2017_by_class` to download selected COCO 2017 classes
and export them into a clear COCO folder layout.

Example:

```bash
python -m scripts.tools.download_coco2017_by_class \
  --output-root data/coco_filtered \
  --classes bowl laptop "cell phone" remote book backpack handbag suitcase toothbrush
```

Expected output layout:

```text
data/coco_filtered/
  train/
    _annotations.coco.json
    *.jpg
  valid/
    _annotations.coco.json
    *.jpg
  test/
    _annotations.coco.json
    *.jpg
```

Notes:

- `validation` is exported as `valid`.
- The final dataset you should use is `train/`, `valid/`, and `test/`.
- The raw COCO files are downloaded into FiftyOne's default cache directory, not under `output-root`.
- On Linux, that cache is commonly located under `~/fiftyone/`.

## Ward dataset preparation(尚未建置)

Download the archive and extract it under `data/`.

Expected location after extraction:

```text
data/ward_dataset/
```

### Linux

```bash
mkdir -p data
wget -O data/ward_dataset.zip \
  https://github.com/chenp6/SegmentationTask/releases/download/add_ward_dataset/ward_dataset.zip
unzip data/ward_dataset.zip -d data/
```

If `wget` is not installed, you can use `curl` instead:

```bash
mkdir -p data
curl -L \
  https://github.com/chenp6/SegmentationTask/releases/download/add_ward_dataset/ward_dataset.zip \
  -o data/ward_dataset.zip
unzip data/ward_dataset.zip -d data/
```

### PowerShell

```powershell
New-Item -ItemType Directory -Force -Path data | Out-Null
Invoke-WebRequest `
  -Uri "https://github.com/chenp6/SegmentationTask/releases/download/add_ward_dataset/ward_dataset.zip" `
  -OutFile "data/ward_dataset.zip"
Expand-Archive -Path "data/ward_dataset.zip" -DestinationPath "data" -Force
```
