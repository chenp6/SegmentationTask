# YOLOv11 Detection Training

這個模組是比照 `scripts/yolov11_seg` 建立的 object detection 版本，
專門使用 `scripts/object_detection_to_image_segmentaion` 原本那份 Roboflow COCO detection 資料集。

預設資料根目錄：

```text
data/hiod_coco/
  train/_annotations.coco.json
  valid/_annotations.coco.json
  test/_annotations.coco.json
```

這套流程會先把 COCO detection 標註轉成 Ultralytics YOLO detection 格式：

```text
data/hiod_coco/yolo_detection/
  train/images
  train/labels
  val/images
  val/labels
  test/images
  test/labels
  data.yaml
```

## 安裝

```bash
pip install ultralytics
```

## 1. 準備資料

```bash
python -m scripts.yolov11_detection.prepare_dataset
```

如果你的原始 detection 資料集不在 `data/hiod_coco`：

```bash
python -m scripts.object_detection_to_image_segmentaion.download_roboflow_coco ^
  --dataset-url https://universe.roboflow.com/{workspace}/{project}/dataset/ ^
  --version {version} ^
  --api-key {api_key} ^
  --output-dir /path/to/hiod_coco

python -m scripts.yolov11_detection.prepare_dataset --data-root path/to/hiod_coco
```

這一步主要是把原始 COCO detection 資料：

- `train/_annotations.coco.json`
- `valid/_annotations.coco.json`
- `test/_annotations.coco.json`

轉成 YOLO 可以直接訓練的：

- `images/*.jpg`
- `labels/*.txt`
- `data.yaml`

## 2. 訓練

最基本的訓練方式：

```bash
python -m scripts.yolov11_detection.train
```

指定模型尺寸：

```bash
python -m scripts.yolov11_detection.train --model yolo11s.pt --epochs 100 --batch-size 16
```

```bash
python -m scripts.yolov11_detection.train --output-dir output/custom_yolov11_detection
```

`--output-dir` 現在就是最終實驗資料夾，訓練結果會直接寫到
`output/custom_yolov11_detection/`，其中包含 `results.csv` 與 `weights/best.pt`。

### `train.py` 目前支援的常用 CLI 參數

- `--data-yaml`
  預設：`None`
  不指定時，程式會依 `--data-root` 或 `config.py` 自動建立 / 使用 YOLO `data.yaml`
- `--data-root`
  預設：`None`
  不指定時，會回退到 `config.py` 內的 `data/hiod_coco`
- `--model`
  預設：`None`
  不指定時，會回退到 `config.py` 內的 `yolo11n.pt`
- `--epochs`
  預設：`None`
  不指定時，會回退到 `config.py` 內的 `100`
- `--batch-size`
  預設：`None`
  不指定時，會回退到 `config.py` 內的 `16`
- `--imgsz`
  預設：`None`
  不指定時，會回退到 `config.py` 內的 `640`
- `--workers`
  預設：`None`
  不指定時，會回退到 `config.py` 內的 `0`
- `--from-scratch`
  預設：關閉
  不加這個參數時，預設會從 pretrained 權重開始 fine-tune

### 常見參數調整範例

使用較大的模型：

```bash
python -m scripts.yolov11_detection.train ^
  --model yolo11s.pt
```

```bash
python -m scripts.yolov11_detection.train ^
  --output-dir output/custom_yolov11_detection
```

增加訓練回合：

```bash
python -m scripts.yolov11_detection.train ^
  --epochs 150
```

修改 batch size：

```bash
python -m scripts.yolov11_detection.train ^
  --batch-size 8
```

修改 resize 輸入尺寸：

```bash
python -m scripts.yolov11_detection.train ^
  --imgsz 960
```

提高 dataloader workers：

```bash
python -m scripts.yolov11_detection.train ^
  --workers 4
```

指定自己的資料根目錄：

```bash
python -m scripts.yolov11_detection.train ^
  --data-root path/to/hiod_coco
```

直接指定已存在的 `data.yaml`：

```bash
python -m scripts.yolov11_detection.train ^
  --data-yaml data/hiod_coco/yolo_detection/data.yaml
```

從 scratch 訓練：

```bash
python -m scripts.yolov11_detection.train ^
  --from-scratch
```

### Data Augmentation 參數範例

目前 `train.py` 已支援從 CLI 傳入這批 augmentation 參數：

```bash
python -m scripts.yolov11_detection.train ^
  --epochs 100 ^
  --batch-size 16 ^
  --imgsz 640 ^
  --hsv-h 0.015 ^
  --hsv-s 0.3 ^
  --hsv-v 0.15 ^
  --degrees 15 ^
  --translate 0.1 ^
  --scale 0.25 ^
  --shear 10 ^
  --perspective 0.0 ^
  --flipud 0.1 ^
  --fliplr 0.5 ^
  --mosaic 0.5 ^
  --mixup 0.05
```

若想先用比較保守的 baseline，可以從這組開始：

```bash
python -m scripts.yolov11_detection.train ^
  --epochs 100 ^
  --batch-size 16 ^
  --imgsz 640 ^
  --hsv-h 0.015 ^
  --hsv-s 0.3 ^
  --hsv-v 0.15 ^
  --degrees 10 ^
  --translate 0.1 ^
  --scale 0.2 ^
  --shear 3 ^
  --perspective 0.0 ^
  --flipud 0.0 ^
  --fliplr 0.5 ^
  --mosaic 0.5 ^
  --mixup 0.0
```

## 3. 評估

```bash
python -m scripts.yolov11_detection.evaluate ^
  --model output/yolov11_detection/exp/weights/best.pt ^
  --data data/hiod_coco/yolo_detection/data.yaml
```

## 4. 視覺化

### 視覺化 prediction result

```bash
python -m scripts.yolov11_detection.visualize ^
  --model output/yolov11_detection/exp/weights/best.pt ^
  --source data/hiod_coco/test ^
  --max-images 10 ^
  --save
```

### 視覺化 ground truth

```bash
python -m scripts.yolov11_detection.visualize ^
  --ground-truth ^
  --data-yaml data/hiod_coco/yolo_detection/data.yaml ^
  --split val ^
  --max-images 10 ^
  --save
```

## Resize 說明

在你目前這套 `yolov11_detection` 流程裡：

- `train` 時，Ultralytics 會自動 resize / letterbox 影像，並同步處理 bbox
- `val/test` 時，也會自動做相同類型的尺寸調整
- 所以你不需要自己手動修改原始 annotation

如果你想改 resize，大致有兩種方式：

- 改 [config.py](/c:/Users/B50137/Downloads/chenp6/SegmentationTask/scripts/yolov11_detection/config.py) 的 `image_size`
- 或訓練時直接加 `--imgsz`

例如：

```bash
python -m scripts.yolov11_detection.train --imgsz 960
```

只有在你先離線把原始圖片改尺寸並存成新資料集時，才需要同步修改 annotation。

## 檔案說明

- `config.py`: 預設資料路徑、YOLOv11 model、訓練參數
- `dataset.py`: COCO detection -> YOLO detection labels 轉換
- `prepare_dataset.py`: 只做資料準備
- `train.py`: 訓練 YOLOv11 detection
- `evaluate.py`: 評估 box mAP
- `visualize.py`: prediction / ground-truth 視覺化
