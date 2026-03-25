# YOLOv11 Detection To Segmentation Evaluation

This workflow evaluates a YOLOv11 detection model on a COCO segmentation dataset by
using each predicted bbox as a SAM2 box prompt, then running COCO `segm` evaluation.

這個流程會把 YOLOv11 detection model 的預測框當作 SAM2 的 box prompt，
在 COCO segmentation dataset 上產生預測 mask，最後使用 COCO `segm` 指標進行評估。

## Basic Command

```bash
python -m scripts.yolov11_detection.evaluate_segmentation \
  --model output/medbin_dataset/yolov11_detection/exp/weights/best.pt \
  --input-root data/medbin_dataset \
  --split test
```

## Optional JSON Export

```bash
python -m scripts.yolov11_detection.evaluate_segmentation \
  --model output/medbin_dataset/yolov11_detection/exp/weights/best.pt \
  --input-root data/medbin_dataset \
  --split test \
  --pred-json-out output/yolov11_detection/segmentation_predictions_test.json
```

## Arguments

### `--model`

English:
Path to the trained YOLOv11 detection weights file.

中文：
訓練完成的 YOLOv11 detection 權重檔路徑。

Example:

```bash
--model output/yolov11_detection/exp/weights/best.pt
```

### `--input-root`

English:
Root directory of the COCO segmentation dataset. The script expects
`<input-root>/<split>/_annotations.coco.json` and the corresponding images under
`<input-root>/<split>/`.

中文：
COCO segmentation dataset 的根目錄。程式會預期資料結構中包含
`<input-root>/<split>/_annotations.coco.json`，以及位於
`<input-root>/<split>/` 之下的對應影像。

Example:

```bash
--input-root data/medbin_dataset
```

### `--split`

English:
Dataset split to evaluate. Common values are `train`, `valid`, or `test`.
Default is `test`.

中文：
要評估的資料切分。常見值為 `train`、`valid` 或 `test`。
預設值為 `test`。

Example:

```bash
--split test
```

### `--conf`

English:
YOLO confidence threshold used during detection inference. Lower values keep more
detections; higher values keep only more confident detections.

中文：
YOLO detection 推論時使用的 confidence threshold。值越低會保留更多框；
值越高則只保留信心較高的框。

Example:

```bash
--conf 0.25
```

### `--iou`

English:
YOLO NMS IoU threshold. This affects how overlapping detection boxes are suppressed.

中文：
YOLO 的 NMS IoU threshold。這個值會影響重疊 detection boxes 的抑制方式。

Example:

```bash
--iou 0.7
```

### `--max-det`

English:
Maximum number of detections kept per image after YOLO inference.

中文：
每張圖片在 YOLO 推論後最多保留的 detection 數量。

Example:

```bash
--max-det 100
```

### `--imgsz`

English:
YOLO inference image size. Larger values may improve small-object detection but
increase computation time and memory usage.

中文：
YOLO 推論時的輸入影像尺寸。較大的值可能有助於小物件偵測，但也會增加運算時間與記憶體使用量。

Example:

```bash
--imgsz 640
```

### `--device`

English:
Execution device, such as `cuda` or `cpu`.

中文：
執行裝置，例如 `cuda` 或 `cpu`。

Example:

```bash
--device cuda
```

### `--sam2-model-id`

English:
SAM2 model identifier used when loading the segmentation model from the installed package.

中文：
透過已安裝套件載入 segmentation model 時所使用的 SAM2 model identifier。

Example:

```bash
--sam2-model-id facebook/sam2.1-hiera-large
```

### `--sam2-checkpoint`

English:
Optional local checkpoint path for SAM2. Use this when a local checkpoint should be
loaded instead of the default model resolution path.

中文：
SAM2 的本機 checkpoint 路徑，非必要。若要改用本機權重而不是預設模型路徑，可使用此參數。

Example:

```bash
--sam2-checkpoint checkpoints/sam2.1_hiera_large.pt
```

### `--sam2-config`

English:
Optional local SAM2 config path. Useful when the model should be loaded from a
specific local config file.

中文：
SAM2 的本機設定檔路徑，非必要。當需要指定本機 config 載入模型時可使用。

Example:

```bash
--sam2-config configs/sam2.1_hiera_l.yaml
```

### `--box-pad-ratio`

English:
Ratio used to expand the predicted bbox before sending it to SAM2. This can help
capture object boundaries that extend slightly outside the YOLO box.

中文：
送進 SAM2 前，用來擴張預測 bbox 的比例。當物件邊界略微超出 YOLO box 時，這有助於提升 mask 覆蓋範圍。

Example:

```bash
--box-pad-ratio 0.05
```

### `--segmentation-format`

English:
Intermediate segmentation format returned by `build_segmentation`. Supported values are
`polygon` and `rle`. The exported COCO prediction still stores an RLE mask for evaluation.

中文：
`build_segmentation` 產生的中間 segmentation 格式，可選 `polygon` 或 `rle`。
不過最後送進 COCO evaluation 的 prediction 仍會以 RLE mask 儲存。

Example:

```bash
--segmentation-format polygon
```

### `--empty-mask-policy`

English:
Fallback behavior when SAM2 does not produce a valid mask or the mask area is too small.
`box` uses a rectangular box mask; `skip` drops that prediction.

中文：
當 SAM2 沒有產生有效 mask，或 mask 面積太小時的處理方式。
`box` 會改用矩形 box mask；`skip` 會直接忽略該筆 prediction。

Example:

```bash
--empty-mask-policy box
```

### `--min-mask-area`

English:
Minimum valid predicted mask area. Masks smaller than this threshold trigger the
empty-mask policy.

中文：
最小有效 predicted mask 面積。若 mask 面積小於此門檻，就會套用 empty-mask policy。

Example:

```bash
--min-mask-area 16
```

### `--max-images`

English:
Limit evaluation to the first N images. Useful for smoke tests, debugging, or
quick pipeline checks.

中文：
只評估前 N 張圖片。適合用於 smoke test、除錯或快速檢查整體流程。

Example:

```bash
--max-images 20
```

### `--single-mask`

English:
Disable SAM2 multi-mask output. When enabled, the script requests a single mask
prediction from SAM2.

中文：
關閉 SAM2 的 multi-mask 輸出。啟用後，程式只會向 SAM2 要求單一 mask prediction。

Example:

```bash
--single-mask
```

### `--pred-json-out`

English:
Optional output path for a per-instance prediction JSON file. This file stores
detailed prediction records such as `image_id`, `category_id`, `score`, `bbox`,
`area`, and `segmentation`.

中文：
每筆 instance prediction 明細 JSON 的輸出路徑，非必要。這份檔案會保存
`image_id`、`category_id`、`score`、`bbox`、`area`、`segmentation`
等詳細資訊。

Example:

```bash
--pred-json-out output/yolov11_detection/segmentation_predictions_test.json
```

## Notes

- `--input-root` should point to a COCO instance-segmentation dataset root.
- The script reads images from `input-root/<split>/`.
- Ground truth comes from `input-root/<split>/_annotations.coco.json`.
- The evaluation assumes preserved category ids are used as YOLO class indices.
- `sam2` must be installed before running this script.

中文補充：

- `--input-root` 應指向 COCO instance-segmentation dataset 的根目錄。
- 程式會從 `input-root/<split>/` 讀取影像。
- Ground truth 來自 `input-root/<split>/_annotations.coco.json`。
- 這支評估程式預設假設 YOLO class index 保留原始 category id。
- 執行前需要先安裝 `sam2`。
