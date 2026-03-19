# Instance PNG Export

This folder now includes a third step for exporting each COCO segmentation instance
as an individual transparent PNG.

## Full Pipeline

### 1. Download Roboflow COCO detection dataset

```bash
python -m scripts.object_detection_to_image_segmentaion.download_roboflow_coco \
  --dataset-url https://universe.roboflow.com/{workspace}/{project} \
  --version {version} \
  --api-key {api_key} \
  --output-dir data/hiod_coco
```

### 2. Convert detection COCO to segmentation COCO

```bash
python -m scripts.object_detection_to_image_segmentaion.convert_coco_detection_to_segmentation \
  --input-root data/hiod_coco \
  --output-root data/hiod_sam2_seg \
  --sam2-model-id facebook/sam2.1-hiera-large \
  --device cuda
```

### 3. Export each segmentation instance to transparent PNG

```bash
python -m scripts.object_detection_to_image_segmentaion.extract_coco_instances_to_png \
  --input-root data/hiod_sam2_seg \
  --output-root output/hiod_instances_png \
  --padding 8
```

## Output Layout

```text
output/hiod_instances_png/
  train/<category_name>/*.png
  valid/<category_name>/*.png
  test/<category_name>/*.png
  manifest.jsonl
```

Each PNG:
- keeps only one instance
- removes the background by writing transparency into the alpha channel
- is cropped to the mask bounding box

## Optional Flags

```bash
--padding 8
--min-mask-area 16
--save-full-mask
```

- `--padding`: add extra pixels around the cropped instance
- `--min-mask-area`: skip very tiny instances
- `--save-full-mask`: also write one full-size binary mask PNG per instance
