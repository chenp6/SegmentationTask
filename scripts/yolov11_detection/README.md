Example pipeline for YOLOv11 object detection.
YOLOv11 物件偵測範例流程。

This workflow includes:
1. Downloading a Roboflow dataset in COCO format
2. Preparing the dataset for YOLO detection training
3. Training a YOLOv11 detection model
4. a. Evaluating the trained model b.Combining YOLO detection with SAM2 for segmentation evaluation
5. Visualizing predictions or YOLO ground truth

此流程包含：
1. 從 Roboflow 下載 COCO 格式資料集
2. 準備 YOLO detection 訓練資料
3. 訓練 YOLOv11 detection 模型
4. a.評估訓練完成的Object detection模型 b.串接 SAM2 進行 segmentation 評估
5. 視覺化 prediction 或 YOLO ground truth


# Step 1. Download dataset from Roboflow:
Step 1. 從 Roboflow 下載資料集：
```bash
    python -m scripts.tools.download_roboflow_coco \
      --dataset-url <ROBOFLOW_DATASET_URL> \
      --version <DATASET_VERSION> \
      --api-key <ROBOFLOW_API_KEY> \
      --output-dir data/<DATASET_NAME>
```
# Step 2. Prepare YOLO detection dataset:
Step 2. 準備 YOLO detection 資料集：
```bash
    python -m scripts.yolov11_detection.prepare_dataset \
      --data-root data/<DATASET_NAME> \
      --output-root data/<DATASET_NAME>/yolo_detection
```
# Step 3. Train YOLOv11 detection model:
Step 3. 訓練 YOLOv11 detection 模型：
```bash
    python -m scripts.yolov11_detection.train \
      --epochs <EPOCHS> \
      --data-root data/<YOLO_DATASET_ROOT> \
      --batch-size <BATCH_SIZE> \
      --imgsz <IMAGE_SIZE> \
      --output-dir output/<EXPERIMENT_NAME>/yolov11_detection \
      --workers <NUM_WORKERS>
```
# Step 4.a. Evaluate trained model:
Step 4.a. 評估訓練完成的模型：
```bash
    python -m scripts.yolov11_detection.evaluate \
      --model <MODEL_PATH> \
      --data <DATA_YAML_PATH> \
      --imgsz <IMAGE_SIZE>
```
# Step 4.b. (For Segmentation) Combine with SAM2 and evaluate the trained model
Step 4.b.（用於分割評估）結合 SAM2 來評估已訓練模型
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```
```bash
python -m scripts.yolov11_detection.evaluate_segmentation \
  --model <MODEL_PATH> \
  --sam2-model-id <SAM2_VERSION> \
  --input-root <DATASET_ROOT> \
  --imgsz <IMAGE_SIZE>
```

**Optional JSON Export**

```bash
python -m scripts.yolov11_detection.evaluate_segmentation \
  --model <MODEL_PATH> \
  --input-root <DATASET_ROOT> \
  --sam2-model-id <SAM2_VERSION> \
  --imgsz <IMAGE_SIZE> \
  --pred-json-out <OUTPUT_JSON_PATH>
```
**Example**
```bash
python -m scripts.yolov11_detection.evaluate_segmentation \
  --model output/yolov11_detection/exp/weights/best.pt \
  --input-root data/midbin_dataset \
  --sam2-model-id facebook/sam2.1-hiera-large \
  --imgsz 640
```


# Step 5. Visualize predictions or YOLO ground truth:
Step 5. 視覺化 prediction 或 YOLO ground truth：

This script is mainly used to quickly inspect whether predicted boxes and class labels
look reasonable, or whether the converted YOLO labels are aligned with the images.
快速檢查 prediction bbox 與類別是否合理，或確認轉換後的 YOLO label 是否和影像正確對齊。

Prediction mode:
Prediction 模式：
```bash
python -m scripts.yolov11_detection.visualize \
  --model output/yolov11_detection/exp/weights/best.pt \
  --source data/<DATASET_NAME>/yolo_detection/test/images \
  --max-images 10 ^
  --save
```

Ground-truth mode:
Ground-truth 模式：
```bash
python -m scripts.yolov11_detection.visualize \
  --ground-truth \
  --data-yaml data/<DATASET_NAME>/yolo_detection/data.yaml \
  --split test \
  --max-images 10 \
  --save \
  --output-dir output/yolov11_detection_gt
```
Notes:
- Prediction mode uses the trained YOLO model to draw predicted boxes on images.
- Ground-truth mode does not load model weights. It reads `data.yaml` and YOLO label `.txt` files, then draws the annotated boxes directly from the dataset.
- `--save` writes rendered images to disk.
- `--show` opens an OpenCV window for interactive viewing.

Notes:
注意事項：
- Prediction mode uses the trained YOLO model to draw predicted boxes on images.
- Prediction 模式會用訓練好的 YOLO 模型在圖片上畫出預測框。
- Ground-truth mode does not load model weights. It reads `data.yaml` and YOLO label `.txt`
  files, then draws the annotated boxes directly from the dataset.
- Ground-truth 模式不會載入模型權重，而是直接讀取 `data.yaml` 和 YOLO label `.txt`
  檔，把資料集標註框畫在圖片上。
- `--save` writes rendered images to disk.
- `--show` opens an OpenCV window for interactive viewing.
- `--save` 會把結果圖片輸出到磁碟。
- `--show` 會開啟 OpenCV 視窗做互動式檢視。

