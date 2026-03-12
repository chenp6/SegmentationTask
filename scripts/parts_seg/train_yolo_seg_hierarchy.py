"""
Train YOLOv11-seg (yolo11n-seg.pt / yolo11s-seg.pt, not YOLO26) with hierarchy.

234 part classes + 44 object (auxiliary) classes. Uses YOLOv11-seg as backbone, custom head
(SegmentWithObject), and dataset that loads .hierarchy.json so each instance has object_id.
Loss = part (box/seg/cls/dfl) + object CE at positives.

Requires: export with --preserve_hierarchy so labels/*.hierarchy.json exist.

Usage (from repo root):
  # 1. Export with hierarchy
  python -m scripts.parts_seg.export_yolo_seg_format --data_root data/ADE20KPart234 --output_dir data/ADE20KPart234_yolo_seg --preserve_hierarchy

  # 2. Train
  python -m scripts.parts_seg.train_yolo_seg_hierarchy --data_yaml data/ADE20KPart234_yolo_seg/data.yaml --epochs 30 --batch 4
"""
import types
from pathlib import Path

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import DEFAULT_CFG, RANK

from .dataset_yolo_seg_hierarchy import YOLOSegmentHierarchyDataset
from .loss_hierarchy import v8SegmentationLossWithObject
from .model_yolo_seg_hierarchy import build_yolo_seg_hierarchy_model


def _hierarchy_init_criterion(model):
    """Module-level so the model (and EMA copy) remain picklable when saving checkpoint."""
    gain = getattr(model, "_object_loss_gain", 0.5)
    return v8SegmentationLossWithObject(model, object_loss_gain=gain)


class HierarchySegmentationTrainer(SegmentationTrainer):
    """Segment trainer with hierarchy dataset and object auxiliary loss."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment"
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss", "object_loss")

    def get_model(self, cfg=None, weights=None, verbose=True):
        data = getattr(self, "data", None)
        if data is None:
            raise RuntimeError("get_model called before get_dataset; ensure data is set")
        nc_part = int(data.get("nc", 234))
        no_object = 44
        # weights may be a checkpoint dict from load_checkpoint(); use path for loading
        checkpoint_path = self.args.model if isinstance(weights, dict) else (weights or self.args.model)
        # Build YOLO wrapper then return inner SegmentationModel (trainer expects inner model)
        yolo_model = build_yolo_seg_hierarchy_model(
            checkpoint=checkpoint_path,
            nc_part=nc_part,
            no_object=no_object,
            device=self.device,
        )
        inner = yolo_model.model
        inner._object_loss_gain = getattr(self, "_object_loss_gain", 0.5)
        inner.init_criterion = types.MethodType(_hierarchy_init_criterion, inner)
        if verbose and RANK == -1:
            print(f"Hierarchy model: nc_part={nc_part}, no_object={no_object}")
        return inner

    def build_dataset(self, img_path, mode="train", batch=None):
        from ultralytics.utils import colorstr
        from ultralytics.utils.torch_utils import unwrap_model

        gs = max(int(unwrap_model(self.model).stride.max()), 32) if self.model else 32
        return YOLOSegmentHierarchyDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or (mode == "val"),
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv11-seg with hierarchy (part + object auxiliary)")
    parser.add_argument("--data_yaml", type=str, required=True, help="Path to data.yaml (from export with --preserve_hierarchy)")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 avoids pin_memory connection errors)")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--project", type=str, default="output/parts_yolo_seg_hierarchy")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--object_loss_gain", type=float, default=0.5, help="Weight for object auxiliary CE")
    args = parser.parse_args()

    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}. Export with --preserve_hierarchy first.")

    # Use absolute project path so Ultralytics saves to output/.../weights/ (not runs/segment/output/...)
    project_path = Path(args.project).resolve()
    overrides = {
        "model": args.model,
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device or None,
        "project": str(project_path),
        "name": args.name,
    }
    trainer = HierarchySegmentationTrainer(cfg=DEFAULT_CFG, overrides=overrides)
    trainer._object_loss_gain = args.object_loss_gain  # keep off trainer.args so validator get_cfg() does not reject it
    trainer.train()
    print("Training done.", trainer.save_dir)


if __name__ == "__main__":
    main()
