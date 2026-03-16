# Object Detection To Image Segmentation

這個資料夾提供一條完整流程，將 Roboflow 的 object detection COCO 資料集，利用現有 bbox 搭配 pretrained SAM2，自動轉成 COCO instance segmentation 資料集。

適合你的情境：
- 資料集原本只有 detection bbox
- 想先用 SAM2 自動生成 mask
- 之後再拿生成的 segmentation dataset 去訓練 YOLO-seg、Mask2Former 或其他 segmentation model

## 流程概念

1. 從 Roboflow 下載 COCO detection 資料集
2. 逐張圖讀取 bbox
3. 用 bbox 當作 SAM2 prompt
4. 讓 SAM2 產生每個物件的 mask
5. 將結果輸出成新的 COCO segmentation 標註

## 建議安裝

```bash
pip install torch torchvision
pip install pillow numpy opencv-python tqdm pycocotools
pip install git+https://github.com/facebookresearch/sam2.git
```

如果你想直接從 Hugging Face 載入預訓練 SAM2，也建議安裝：

```bash
pip install transformers huggingface_hub
```

## Step 1: 下載 Roboflow COCO detection 資料集

你的資料集頁面：

```text
https://universe.roboflow.com/myworkspacename/hiod-1fqtj
```

下載時你還需要知道 Roboflow dataset version，指令範例：

```bash
python -m scripts.object_detection_to_image_segmentaion.download_roboflow_coco \
  --dataset-url https://universe.roboflow.com/myworkspacename/hiod-1fqtj \
  --version 1 \
  --api-key <ROBOFLOW_API_KEY> \
  --output-dir data/hiod_coco
```

也可以把 API key 放在環境變數：

```bash
set ROBOFLOW_API_KEY=your_key_here
```

或放在專案根目錄的 `roboflow_credentials.json`。

下載後輸入資料夾通常會長這樣：

```text
data/hiod_coco/
├─ train/
│  ├─ _annotations.coco.json
│  └─ *.jpg
├─ valid/
│  ├─ _annotations.coco.json
│  └─ *.jpg
└─ test/
   ├─ _annotations.coco.json
   └─ *.jpg
```

## Step 2: 用 SAM2 將 detection 轉成 segmentation

最基本的轉換指令：

```bash
python -m scripts.object_detection_to_image_segmentaion.convert_coco_detection_to_segmentation \
  --input-root data/hiod_coco \
  --output-root data/hiod_sam2_seg \
  --sam2-model-id facebook/sam2.1-hiera-large \
  --device cuda
```

建議實際使用版本：

```bash
python -m scripts.object_detection_to_image_segmentaion.convert_coco_detection_to_segmentation \
  --input-root data/hiod_coco \
  --output-root data/hiod_sam2_seg \
  --sam2-model-id facebook/sam2.1-hiera-large \
  --device cuda \
  --box-pad-ratio 0.02 \
  --segmentation-format polygon \
  --empty-mask-policy box
```

輸出會是新的 COCO segmentation dataset：

```text
data/hiod_sam2_seg/
├─ train/
│  ├─ _annotations.coco.json
│  └─ *.jpg
├─ valid/
│  ├─ _annotations.coco.json
│  └─ *.jpg
└─ test/
   ├─ _annotations.coco.json
   └─ *.jpg
```

## 重要參數

- `--sam2-model-id`: 預訓練 SAM2 模型，預設 `facebook/sam2.1-hiera-large`
- `--sam2-checkpoint`: 本地 SAM2 checkpoint
- `--sam2-config`: 本地 SAM2 config
- `--device`: `cuda` 或 `cpu`
- `--box-pad-ratio`: 在原 bbox 外額外擴張 prompt 區域
- `--segmentation-format`: `polygon` 或 `rle`
- `--empty-mask-policy`: `box` 或 `skip`
- `--max-images`: 只跑前 N 張，適合 smoke test
- `--copy-images`: 預設會優先 hard-link；加這個參數就改成直接 copy

## 建議工作流

第一次先小量測試：

```bash
python -m scripts.object_detection_to_image_segmentaion.convert_coco_detection_to_segmentation \
  --input-root data/hiod_coco \
  --output-root data/hiod_sam2_seg_debug \
  --device cuda \
  --max-images 20
```

確認 `_annotations.coco.json` 和 mask 品質沒問題，再跑完整資料集。

## 注意事項

- 用 bbox prompt 產生的 mask 很適合當 bootstrap dataset，但通常還是建議抽樣人工檢查
- 若某些框太鬆或遮擋嚴重，SAM2 可能失敗；目前腳本可選擇回退成 box polygon 或直接跳過
- 如果你後續要接 Ultralytics YOLO-seg，`polygon` 格式通常會比 `rle` 更方便
