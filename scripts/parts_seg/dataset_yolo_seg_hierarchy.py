"""
YOLO-seg dataset with hierarchy: loads .hierarchy.json and adds object_id per instance
so the batch includes object_id (same length as cls) for auxiliary object loss.

Use with Ultralytics-style training when data was exported with --preserve_hierarchy.
"""
from pathlib import Path
import json

import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import img2label_paths


class YOLOSegmentHierarchyDataset(YOLODataset):
    """Segment dataset that adds object_id per instance from .hierarchy.json sidecars."""

    def get_labels(self):
        labels = super().get_labels()
        self.label_files = img2label_paths(self.im_files)
        for i, lb in enumerate(labels):
            n_inst = len(lb["cls"])
            if n_inst == 0:
                lb["object_ids"] = []
                continue
            label_path = Path(self.label_files[i])
            hier_path = label_path.with_name(label_path.stem + ".hierarchy.json")
            if hier_path.exists():
                try:
                    with open(hier_path, encoding="utf-8") as f:
                        hier = json.load(f)
                    # one entry per line in .txt
                    lb["object_ids"] = [int(h["object_id"]) for h in hier]
                    if len(lb["object_ids"]) != n_inst:
                        lb["object_ids"] = [0] * n_inst
                except (json.JSONDecodeError, KeyError):
                    lb["object_ids"] = [0] * n_inst
            else:
                lb["object_ids"] = [0] * n_inst
        return labels

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        lb = self.labels[index]
        object_ids = lb.get("object_ids", [])
        if object_ids:
            batch["object_id"] = torch.tensor(object_ids, dtype=torch.long)
        else:
            batch["object_id"] = torch.zeros(0, dtype=torch.long)
        return batch

    @staticmethod
    def collate_fn(batch):
        new_batch = YOLODataset.collate_fn(batch)
        if batch and "object_id" in batch[0]:
            new_batch["object_id"] = torch.cat([b["object_id"] for b in batch], 0)
        return new_batch

