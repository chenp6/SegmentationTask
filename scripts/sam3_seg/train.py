"""
Train SAM3 query-based instance segmentation on the hospital COCO dataset.

Uses Hungarian matching (scipy) to assign queries to GT instances, then trains with:
  1. Classification CE (all queries)
  2. Mask BCE + Dice (matched queries only)

Usage:
    python -m scripts.sam3_seg.train
    python -m scripts.sam3_seg.train --epochs 20 --batch_size 1
"""
import argparse
import contextlib
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AugConfig, DataConfig, ModelConfig, TrainConfig
from .dataset import HospitalCOCOSegDataset
from .segmentation_model import SAM3SegModel


# --------------------------------------------------------------------------- #
# Collate                                                                      #
# --------------------------------------------------------------------------- #

def collate_fn(batch, mask_size=128):
    """Collate for query training. Pre-resizes GT masks to decoder resolution."""
    images = torch.stack([item["image"] for item in batch])
    targets = []
    for item in batch:
        masks, labels = [], []
        for m, l in zip(item["instance_masks"], item["class_labels"]):
            mt = torch.from_numpy(m).float()
            if mt.sum() > 0:
                masks.append(mt)
                labels.append(l)
        if masks:
            stacked = torch.stack(masks)
            if stacked.shape[-1] != mask_size:
                stacked = F.interpolate(
                    stacked.unsqueeze(1), (mask_size, mask_size),
                    mode="bilinear", align_corners=False
                ).squeeze(1)
            targets.append({"masks": stacked, "labels": torch.tensor(labels, dtype=torch.long)})
        else:
            targets.append({"masks": torch.zeros(0, mask_size, mask_size),
                            "labels": torch.zeros(0, dtype=torch.long)})
    return images, targets


# --------------------------------------------------------------------------- #
# Hungarian Matching                                                           #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def hungarian_match(pred_logits, pred_masks, targets, num_classes):
    """Standard Hungarian matching via scipy."""
    B = pred_logits.shape[0]
    matches = []

    match_size = (32, 32)
    pred_masks_small = F.interpolate(
        pred_masks, size=match_size, mode="bilinear", align_corners=False
    )

    for b in range(B):
        tgt_labels = targets[b]["labels"]
        tgt_masks = targets[b]["masks"]
        n_gt = len(tgt_labels)

        if n_gt == 0:
            matches.append((torch.tensor([], dtype=torch.long),
                            torch.tensor([], dtype=torch.long)))
            continue

        tgt_masks_small = F.interpolate(
            tgt_masks.unsqueeze(0).to(pred_masks.device),
            size=match_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        # Class cost
        pred_prob = pred_logits[b].softmax(-1)
        tgt_labels_dev = tgt_labels.to(pred_prob.device)
        class_cost = -pred_prob[:, tgt_labels_dev]

        # Dice cost
        pred_flat = pred_masks_small[b].sigmoid().flatten(1)
        tgt_flat = tgt_masks_small.flatten(1)
        inter = (pred_flat.unsqueeze(1) * tgt_flat.unsqueeze(0)).sum(-1)
        union = pred_flat.unsqueeze(1).sum(-1) + tgt_flat.unsqueeze(0).sum(-1)
        dice_cost = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)

        cost = 2.0 * class_cost + 5.0 * dice_cost
        cost_np = cost.cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost_np)
        matches.append((torch.tensor(row_ind, dtype=torch.long),
                        torch.tensor(col_ind, dtype=torch.long)))

    return matches


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #

def compute_loss(pred_logits, pred_masks, targets, matches, num_classes, cfg):
    """Query-based loss: class CE + mask BCE + mask Dice."""
    device = pred_logits.device
    B, N, _ = pred_logits.shape
    no_obj_class = num_classes

    total_cls = torch.tensor(0.0, device=device)
    total_bce = torch.tensor(0.0, device=device)
    total_dice = torch.tensor(0.0, device=device)
    n_matched = 0

    cls_weight = torch.ones(num_classes + 1, device=device)
    cls_weight[no_obj_class] = cfg.no_object_weight
    dec_h, dec_w = pred_masks.shape[2:]

    for b in range(B):
        query_idx, gt_idx = matches[b]
        tgt_labels = targets[b]["labels"].to(device)
        tgt_masks = targets[b]["masks"]

        cls_target = torch.full((N,), no_obj_class, dtype=torch.long, device=device)
        if len(query_idx) > 0:
            cls_target[query_idx] = tgt_labels[gt_idx]

        total_cls = total_cls + F.cross_entropy(pred_logits[b], cls_target, weight=cls_weight)

        if len(query_idx) == 0:
            continue

        if tgt_masks.shape[-1] != dec_w or tgt_masks.shape[-2] != dec_h:
            tgt_masks_resized = F.interpolate(
                tgt_masks.unsqueeze(0).to(device), (dec_h, dec_w),
                mode="bilinear", align_corners=False
            ).squeeze(0)
        else:
            tgt_masks_resized = tgt_masks.to(device)

        pred_matched = pred_masks[b, query_idx]
        tgt_matched = tgt_masks_resized[gt_idx]

        total_bce = total_bce + F.binary_cross_entropy_with_logits(
            pred_matched, tgt_matched, reduction="mean"
        )

        pred_sig = pred_matched.sigmoid()
        inter = (pred_sig * tgt_matched).sum((-1, -2))
        union = pred_sig.sum((-1, -2)) + tgt_matched.sum((-1, -2))
        dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
        total_dice = total_dice + dice.mean()

        n_matched += len(query_idx)

    total_cls /= B
    total_bce /= max(1, B)
    total_dice /= max(1, B)

    total = cfg.cls_weight * total_cls + cfg.mask_bce_weight * total_bce + cfg.mask_dice_weight * total_dice

    return total, {
        "cls": total_cls.item(), "bce": total_bce.item(),
        "dice": total_dice.item(), "total": total.item(), "matched": n_matched,
    }


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

def train(data_cfg=None, model_cfg=None, train_cfg=None, aug_cfg=None, resume=None):
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    aug_cfg   = aug_cfg   or AugConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(model_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = HospitalCOCOSegDataset(
        data_cfg.data_root, split="train", image_size=data_cfg.image_size, augment=True
    )
    val_dataset = HospitalCOCOSegDataset(
        data_cfg.data_root, split="valid", image_size=data_cfg.image_size, augment=False
    )
    num_classes = train_dataset.num_classes + 1
    model_cfg.num_classes = num_classes
    print(f"Num classes (incl. background): {num_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Model
    model = SAM3SegModel.build(
        model_name=model_cfg.sam3_model_name,
        checkpoint_path=model_cfg.sam3_checkpoint,
        config_path=model_cfg.sam3_config,
        num_classes=num_classes,
        use_adapters=model_cfg.use_adapters,
        adapter_dim=model_cfg.adapter_dim,
        image_size=data_cfg.image_size,
        decoder_type=model_cfg.decoder_type,
        num_queries=model_cfg.num_queries,
        decoder_layers=model_cfg.decoder_layers,
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        dim_ff=model_cfg.dim_ff,
    )
    model.to(device)

    # Optimizer
    adapter_lr = train_cfg.learning_rate * train_cfg.adapter_lr_factor
    param_groups = model.get_param_groups(adapter_lr=adapter_lr, decoder_lr=train_cfg.learning_rate)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.weight_decay)

    total_steps = max(1, len(train_loader) * train_cfg.num_epochs // train_cfg.grad_accum_steps)
    warmup_steps = train_cfg.lr_warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_amp_ctx():
        if train_cfg.bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        elif train_cfg.fp16:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume and Path(resume).exists():
        print(f"Resuming from: {resume}")
        ckpt = torch.load(os.path.join(resume, "training_state.pt"),
                          map_location=device, weights_only=False)
        current_state = model.state_dict()
        for k, v in ckpt.get("model_state", {}).items():
            if k in current_state and current_state[k].shape == v.shape:
                current_state[k] = v
        model.load_state_dict(current_state)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Training loop
    global_step = 0
    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()
        model.backbone.eval()
        epoch_losses = {"cls": 0, "bce": 0, "dice": 0, "total": 0, "matched": 0}
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)

            with get_amp_ctx():
                outputs = model(images)
                matches = hungarian_match(
                    outputs["pred_logits"], outputs["pred_masks"],
                    targets, num_classes,
                )
                loss, loss_dict = compute_loss(
                    outputs["pred_logits"], outputs["pred_masks"],
                    targets, matches, num_classes, train_cfg,
                )
                loss = loss / train_cfg.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % train_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            pbar.set_postfix(total=f"{loss_dict['total']:.3f}", matched=loss_dict['matched'])

        n = len(train_loader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        print(
            f"\n  Epoch {epoch+1} — "
            f"total={avg['total']:.4f}  cls={avg['cls']:.4f}  "
            f"bce={avg['bce']:.4f}  dice={avg['dice']:.4f}  "
            f"matched={avg['matched']:.1f}"
        )

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(device)
                with get_amp_ctx():
                    outputs = model(images)
                    matches = hungarian_match(
                        outputs["pred_logits"], outputs["pred_masks"],
                        targets, num_classes,
                    )
                    loss, _ = compute_loss(
                        outputs["pred_logits"], outputs["pred_masks"],
                        targets, matches, num_classes, train_cfg,
                    )
                val_total += loss.item()

        avg_val = val_total / max(1, len(val_loader))
        print(f"  Epoch {epoch+1} val loss: {avg_val:.4f}")

        # Checkpoint
        ckpt_data = {
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "model_name": model_cfg.sam3_model_name,
            "use_adapters": model_cfg.use_adapters,
            "adapter_dim": model_cfg.adapter_dim,
            "image_size": data_cfg.image_size,
            "decoder_type": model_cfg.decoder_type,
            "num_queries": model_cfg.num_queries,
            "decoder_layers": model_cfg.decoder_layers,
            "d_model": model_cfg.d_model,
            "nhead": model_cfg.nhead,
            "dim_ff": model_cfg.dim_ff,
        }

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = os.path.join(model_cfg.output_dir, "best_model")
            Path(best_path).mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_data, os.path.join(best_path, "model.pt"))
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % train_cfg.save_every_epochs == 0:
            ckpt_path = os.path.join(model_cfg.output_dir, f"checkpoint_epoch{epoch+1}")
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            ckpt_data.update({
                "epoch": epoch + 1, "best_val_loss": best_val_loss,
                "optimizer_state": optimizer.state_dict(),
            })
            torch.save(ckpt_data, os.path.join(ckpt_path, "training_state.pt"))
            print(f"  Saved checkpoint: {ckpt_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train_cfg = TrainConfig()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()

    if args.epochs:      train_cfg.num_epochs = args.epochs
    if args.batch_size:  train_cfg.batch_size = args.batch_size
    if args.data_root:   data_cfg.data_root = args.data_root
    if args.image_size:  data_cfg.image_size = args.image_size
    if args.lr:          train_cfg.learning_rate = args.lr

    train(data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, resume=args.resume)
