# YOLOv11 Detection Training

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

若原始 detection 資料集不在 `data/hiod_coco`：

```bash
python -m scripts.yolov11_detection.prepare_dataset --data-root path/to/hiod_coco
```

## 2. 訓練

```bash
python -m scripts.yolov11_detection.train
```

或指定模型尺寸：

```bash
python -m scripts.yolov11_detection.train --model yolo11s.pt --epochs 100 --batch-size 16
```

想把 data augmentation 參數改成從 CLI 傳入，可以採用這種 args 風格：

```bash
python -m scripts.yolov11_detection.train ^
  --model yolo11s.pt ^
  --epochs 100 ^
  --batch-size 16 ^
  --hsv-h 0.015 ^
  --hsv-s 0.7 ^
  --hsv-v 0.4 ^
  --degrees 5 ^
  --translate 0.1 ^
  --scale 0.5 ^
  --shear 2 ^
  --perspective 0.0 ^
  --flipud 0.1 ^
  --fliplr 0.5 ^
  --mosaic 0.8 ^
  --mixup 0.1
```

目前 `scripts.yolov11_detection.train` 還沒有實作這批 augmentation CLI 參數；
現在若要調整 augmentation，請先修改 `config.py` 內的設定值。

## 3. 評估

```bash
python -m scripts.yolov11_detection.evaluate \
  --model output/yolov11_detection/exp/weights/best.pt \
  --data data/hiod_coco/yolo_detection/data.yaml
```

## 4. 視覺化
想查看result
```bash
python -m scripts.yolov11_detection.visualize \
  --model output/yolov11_detection/exp/weights/best.pt \
  --source data/hiod_coco/test \
  --max-images 10 \
  --save
```

想查看ground truth
```bash
python -m scripts.yolov11_detection.visualize ^
  --ground-truth ^
  --data-yaml data/hiod_coco/yolo_detection/data.yaml ^
  --split val ^
  --max-images 10 ^
  --save
```

## 檔案說明

- `config.py`: 預設資料路徑、YOLOv11 model、訓練參數
- `dataset.py`: COCO detection -> YOLO detection labels 轉換
- `prepare_dataset.py`: 只做資料準備
- `train.py`: 訓練 YOLOv11 detection
- `evaluate.py`: 驗證 box mAP
- `visualize.py`: 預測視覺化
