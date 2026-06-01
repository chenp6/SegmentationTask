import time

import fiftyone as fo
import fiftyone.zoo as foz
from requests.exceptions import ReadTimeout, RequestException


# classes = [
#     "person",
#     "chair",
#     "couch",
#     "potted plant",
#     "bottle",
#     "cup",
#     "fork",
#     "knife",
#     "spoon",
#     "bowl",
#     "toilet",
#     "tv",
#     "laptop",
#     "cell phone",
#     "microwave",
#     "oven",
#     "refrigerator",
#     "book",
#     "remote",
#     "suitcase",
#     "toothbrush",
#     "handbag",
#     "backpack",
# ]
classes = [
    "bowl",
    "cell phone",   # COCO 類別名稱是 cell phone
    "remote",
    "backpack",
    "handbag",
    "suitcase",
    "toothbrush",
    "laptop",
    "book",
]
# COCO 2017 有標註的是 train / validation
fo.config.dataset_zoo_dir = "/data/chenp6/SegmentationTask/data/fiftyone_zoo_cache"

MAX_RETRIES = 8
BASE_RETRY_WAIT_SECONDS = 8
NUM_WORKERS = 1


for split, out_split in [("train", "train"), ("validation", "valid")]:
    dataset = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split=split,
                classes=classes,
                only_matching=True,
                label_types=["detections", "segmentations"],
                num_workers=NUM_WORKERS,
                dataset_name=f"coco2017_{split}",
            )
            break
        except (ReadTimeout, RequestException, TimeoutError) as e:
            if attempt >= MAX_RETRIES:
                raise

            wait_s = BASE_RETRY_WAIT_SECONDS * (2 ** (attempt - 1))
            print(
                f"[WARN] {split} download failed on attempt {attempt}/{MAX_RETRIES}: {e}. "
                f"Retrying in {wait_s}s..."
            )
            time.sleep(wait_s)

    if dataset is None:
        raise RuntimeError(f"Failed to prepare split '{split}' after {MAX_RETRIES} retries")

    dataset.export(
        export_dir=f"/data/chenp6/SegmentationTask/data/coco2017/{out_split}",
        dataset_type=fo.types.COCODetectionDataset,
    )
