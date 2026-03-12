"""
Train SAM2+YOLO on part segmentation: part instances as boxes + labels + masks.

Datasets: pascal (Pascal-Part), ade20k234 (ADE20K Part in data/ADE20KPart234).

Usage (from repo root):
  python -m scripts.parts_seg.train_yolo
  python -m scripts.parts_seg.train_yolo --dataset ade20k234 --data_root data/ADE20KPart234
  python -m scripts.parts_seg.train_yolo --dataset pascal --data_root data/pascal_part --image_size 512
"""
import argparse
import contextlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from scripts.sam2_yolo_seg.train_yolo import (
        Config as BaseConfig,
        fcos_targets_enhanced,
        compute_losses,
    )
    from scripts.sam2_yolo_seg.models.sam2_yolo_model import SAM2YOLO
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from scripts.sam2_yolo_seg.train_yolo import (
        Config as BaseConfig,
        fcos_targets_enhanced,
        compute_losses,
    )
    from scripts.sam2_yolo_seg.models.sam2_yolo_model import SAM2YOLO

from .dataset import PascalPartDataset, collate_fn as _pascal_collate
from .dataset_ade20k234 import ADE20KPart234Dataset, collate_fn as _ade_collate


class Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.data_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "pascal_part"
        )
        self.data_root = os.path.abspath(self.data_root)
        self.output_dir = "output/parts_seg_yolo"
        self.image_size = 512
        self.num_epochs = 30
        self.batch_size = 4
        self.num_workers = 0  # avoid DataLoader pin_memory multiprocessing connection resets
        self.learning_rate = 1e-4
        self.dataset = "pascal"


def train(cfg: Config = None):
    cfg = cfg or Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    if getattr(cfg, "dataset", "pascal") == "ade20k234":
        train_dataset = ADE20KPart234Dataset(
            cfg.data_root, split="train", image_size=cfg.image_size, augment=True
        )
        val_dataset = ADE20KPart234Dataset(
            cfg.data_root, split="val", image_size=cfg.image_size, augment=False
        )
        collate_fn = _ade_collate
    else:
        train_dataset = PascalPartDataset(
            cfg.data_root, split="train", image_size=cfg.image_size, augment=True
        )
        val_dataset = PascalPartDataset(
            cfg.data_root, split="val", image_size=cfg.image_size, augment=False
        )
        collate_fn = _pascal_collate
    num_classes = train_dataset.num_classes

    # num_workers=0 and pin_memory=False avoid ConnectionResetError in DataLoader multiprocessing
    use_pin_memory = cfg.num_workers > 0 and torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )

    model = SAM2YOLO.build(num_classes=num_classes, image_size=cfg.image_size)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.get_param_groups(cfg.learning_rate), weight_decay=cfg.weight_decay
    )
    total_steps = len(train_loader) * cfg.num_epochs // max(1, cfg.grad_accum_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    def get_amp_ctx():
        if getattr(cfg, "bf16", False):
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if getattr(cfg, "fp16", True):
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    for epoch in range(cfg.num_epochs):
        model.train()
        model.backbone.eval()
        optimizer.zero_grad()
        epoch_losses = {"total": 0, "cls": 0, "bbox": 0, "mask": 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)

            with get_amp_ctx():
                outputs = model(images)
                fpn_shapes = [(lvl["cls"].shape[2], lvl["cls"].shape[3]) for lvl in outputs["levels"]]
                level_targets = fcos_targets_enhanced(
                    targets, fpn_shapes, model.strides, cfg.image_size, num_classes, device
                )
                loss, loss_dict = compute_losses(
                    outputs, targets, level_targets, cfg, num_classes, model.strides, cfg.image_size
                )
                loss = loss / cfg.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            for k in epoch_losses:
                if k in loss_dict:
                    epoch_losses[k] += loss_dict[k]
            pbar.set_postfix(
                total=f"{loss_dict['total']:.2f}",
                cls=f"{loss_dict['cls']:.4f}",
                mask=f"{loss_dict['mask']:.4f}",
            )

            if (batch_idx + 1) % cfg.save_every_n_steps == 0:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": batch_idx + 1,
                        "model_state": model.state_dict(),
                        "num_classes": num_classes,
                        "image_size": cfg.image_size,
                    },
                    os.path.join(cfg.output_dir, "latest_model.pt"),
                )

        n = len(train_loader)
        print(f"Epoch {epoch+1} avg loss: {epoch_losses['total']/n:.4f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "image_size": cfg.image_size,
            },
            os.path.join(cfg.output_dir, "latest_model.pt"),
        )
        if (epoch + 1) % cfg.save_every_epochs == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "image_size": cfg.image_size,
                },
                os.path.join(cfg.output_dir, f"epoch_{epoch+1}.pt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pascal", choices=["pascal", "ade20k234"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (default 0 to avoid pin_memory connection errors)")
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset = args.dataset
    if args.data_root:
        cfg.data_root = os.path.abspath(args.data_root)
    elif args.dataset == "ade20k234":
        cfg.data_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "ADE20KPart234")
        )
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    train(cfg)
