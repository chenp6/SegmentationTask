"""
Train ConvNeXt + prototype instance segmentation (FCOS + YOLACT style).

Assignment: FCOS-style (each GT bbox assigns pixels inside it at appropriate FPN level)
Losses: Focal (cls) + GIoU (bbox) + BCE (mask via prototypes)

Usage:
    python -m scripts.convnext_seg.train
"""
import argparse
import contextlib
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DataConfig, ModelConfig, TrainConfig
from .dataset import HospitalCOCODataset, collate_fn
from .model import ConvNeXtSegModel


# --------------------------------------------------------------------------- #
# FCOS Target Assignment                                                       #
# --------------------------------------------------------------------------- #

def fcos_targets(targets, fpn_shapes, strides, image_size, num_classes, device):
    """
    Assign GT boxes to FPN grid positions (FCOS-style).

    For each FPN level, each spatial position (x, y) is inside zero or more GT boxes.
    We assign it to the smallest GT box that contains it.

    Returns per-level tensors:
        cls_targets: [B, H, W] long (0 = background, 1..N = class)
        bbox_targets: [B, 4, H, W] (l, t, r, b distances to bbox edges)
        centerness_targets: [B, 1, H, W]
        matched_gt_idx: [B, H, W] long (-1 = no match)

    Size ranges per level (FCOS convention):
        P3: [0, 64], P4: [64, 128], P5: [128, inf]
    """
    B = len(targets)
    size_ranges = [(0, 64), (64, 128), (128, 1e6)]  # for P3, P4, P5
    all_level_targets = []

    for level_idx, (feat_h, feat_w) in enumerate(fpn_shapes):
        stride = strides[level_idx]
        lo, hi = size_ranges[level_idx]

        cls_tgt = torch.zeros(B, feat_h, feat_w, dtype=torch.long, device=device)
        bbox_tgt = torch.zeros(B, 4, feat_h, feat_w, dtype=torch.float32, device=device)
        ctr_tgt = torch.zeros(B, 1, feat_h, feat_w, dtype=torch.float32, device=device)
        gt_idx_tgt = torch.full((B, feat_h, feat_w), -1, dtype=torch.long, device=device)

        # Grid positions (pixel coords of each feature map cell center)
        shifts_y = (torch.arange(feat_h, device=device).float() + 0.5) * stride
        shifts_x = (torch.arange(feat_w, device=device).float() + 0.5) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        for b in range(B):
            boxes = targets[b]["boxes"].to(device)   # [N, 4] normalized xyxy
            labels = targets[b]["labels"].to(device)  # [N]
            n_gt = len(boxes)
            if n_gt == 0:
                continue

            # Denormalize boxes to pixel coords
            boxes_px = boxes.clone()
            boxes_px[:, [0, 2]] *= image_size
            boxes_px[:, [1, 3]] *= image_size

            # For each box, compute distances from each grid point to box edges
            # boxes_px: [N, 4] = x1, y1, x2, y2
            for g in range(n_gt):
                x1, y1, x2, y2 = boxes_px[g]
                bw, bh = x2 - x1, y2 - y1
                max_side = max(bw.item(), bh.item())

                # Size filter: only assign to appropriate FPN level
                if max_side < lo or max_side >= hi:
                    continue

                # Compute distances (l, t, r, b) from each grid position
                l = shift_x - x1
                t = shift_y - y1
                r = x2 - shift_x
                b_dist = y2 - shift_y

                # Points inside the box: all distances > 0
                inside = (l > 0) & (t > 0) & (r > 0) & (b_dist > 0)

                if not inside.any():
                    continue

                # Centerness (FCOS paper formula)
                dists = torch.stack([l, t, r, b_dist], dim=0)  # [4, H, W]
                ct = (torch.min(l, r) * torch.min(t, b_dist)) / \
                     (torch.max(l, r) * torch.max(t, b_dist) + 1e-6)
                ct = ct.sqrt()

                # Assign: prefer smaller boxes (overwrite larger)
                # Check if current assignment has a larger box
                already = gt_idx_tgt[b] >= 0
                smaller = inside & (~already | (ct > ctr_tgt[b, 0]))

                if smaller.any():
                    cls_tgt[b][smaller] = labels[g] + 1  # +1 for background=0
                    bbox_tgt[b, 0][smaller] = l[smaller]
                    bbox_tgt[b, 1][smaller] = t[smaller]
                    bbox_tgt[b, 2][smaller] = r[smaller]
                    bbox_tgt[b, 3][smaller] = b_dist[smaller]
                    ctr_tgt[b, 0][smaller] = ct[smaller]
                    gt_idx_tgt[b][smaller] = g

        all_level_targets.append({
            "cls": cls_tgt,
            "bbox": bbox_tgt,
            "centerness": ctr_tgt,
            "gt_idx": gt_idx_tgt,
        })

    return all_level_targets


# --------------------------------------------------------------------------- #
# Losses                                                                       #
# --------------------------------------------------------------------------- #

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for multi-class classification (0 = background)."""
    num_classes = logits.shape[1]
    # logits: [B, C, H, W] → [B*H*W, C]
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    targets_flat = targets.reshape(-1)  # [B*H*W]

    # One-hot encode (exclude background class 0 from target)
    # targets_flat: 0=bg, 1..N=foreground classes
    # logits predict classes 0..N-1 (no background logit — background = low scores)
    # Actually for simplicity, use CE with focal weighting
    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    pt = torch.exp(-ce)
    focal = alpha * (1 - pt) ** gamma * ce

    # Only count loss from positive samples and a fraction of negatives
    pos_mask = targets_flat > 0
    n_pos = max(1, pos_mask.sum().item())

    loss = focal[pos_mask].sum() / n_pos
    # Add small background contribution
    if (~pos_mask).any():
        loss = loss + 0.1 * focal[~pos_mask].mean()

    return loss


def giou_loss(pred_bbox, target_bbox, mask):
    """GIoU loss for bbox regression. Only computed on positive samples."""
    # pred_bbox, target_bbox: [B, 4, H, W] (l, t, r, b distances)
    # mask: [B, H, W] boolean (positive positions)

    if not mask.any():
        return torch.tensor(0.0, device=pred_bbox.device)

    # Extract positive positions: [N_pos, 4]
    pred = pred_bbox.permute(0, 2, 3, 1)[mask]      # [N, 4]
    tgt = target_bbox.permute(0, 2, 3, 1)[mask]      # [N, 4]

    # Convert (l, t, r, b) to areas
    pred_area = (pred[:, 0] + pred[:, 2]) * (pred[:, 1] + pred[:, 3])
    tgt_area = (tgt[:, 0] + tgt[:, 2]) * (tgt[:, 1] + tgt[:, 3])

    # Intersection
    inter_w = torch.min(pred[:, 0], tgt[:, 0]) + torch.min(pred[:, 2], tgt[:, 2])
    inter_h = torch.min(pred[:, 1], tgt[:, 1]) + torch.min(pred[:, 3], tgt[:, 3])
    inter = inter_w.clamp(min=0) * inter_h.clamp(min=0)

    union = pred_area + tgt_area - inter + 1e-6
    iou = inter / union

    # Enclosing box
    enc_w = torch.max(pred[:, 0], tgt[:, 0]) + torch.max(pred[:, 2], tgt[:, 2])
    enc_h = torch.max(pred[:, 1], tgt[:, 1]) + torch.max(pred[:, 3], tgt[:, 3])
    enc_area = enc_w * enc_h + 1e-6

    giou = iou - (enc_area - union) / enc_area
    return (1 - giou).mean()


def mask_loss(protos, coeffs, targets, level_targets, strides, image_size):
    """
    Prototype mask loss: for each positive pixel, compute mask from prototypes
    and compare against GT mask.
    """
    B = protos.shape[0]
    proto_h, proto_w = protos.shape[2:]
    device = protos.device
    total_loss = torch.tensor(0.0, device=device)
    n_pos = 0

    for b in range(B):
        gt_masks = targets[b]["masks"].to(device)  # [N_gt, H_img, W_img]
        if len(gt_masks) == 0:
            continue

        # Resize GT masks to prototype resolution
        gt_masks_small = F.interpolate(
            gt_masks.unsqueeze(1), (proto_h, proto_w),
            mode="bilinear", align_corners=False
        ).squeeze(1)  # [N_gt, proto_h, proto_w]

        # Collect coefficients from positive positions across all levels
        for level_idx, lt in enumerate(level_targets):
            gt_idx = lt["gt_idx"][b]       # [H, W]
            pos_mask = gt_idx >= 0
            if not pos_mask.any():
                continue

            # For simplicity: average coefficients per GT instance
            unique_gts = gt_idx[pos_mask].unique()
            for g in unique_gts:
                g_mask = (gt_idx == g)
                # This level might have this GT's coefficients
                # Get coefficients from the prediction
                # We need per-level coeffs — but we combine across levels
                pass

        # Simpler approach: per-GT, gather coefficients from assigned level
        # Average the coefficients across assigned positions, compute mask loss once
        all_coeffs_by_gt = {}
        for level_idx, lt in enumerate(level_targets):
            gt_idx = lt["gt_idx"][b]
            pos_mask = gt_idx >= 0
            if not pos_mask.any():
                continue

            # Need actual predicted coefficients for this level
            # This is passed separately — for now use a simplified version

        # Simplified: compute mask loss using top-down approach
        # For each GT, find its assigned level and positions

    return total_loss / max(1, n_pos)


def compute_losses(model_output, targets, level_targets, cfg, num_classes):
    """Compute all losses: focal cls + GIoU bbox + centerness + mask."""
    device = model_output["protos"].device

    total_cls = torch.tensor(0.0, device=device)
    total_bbox = torch.tensor(0.0, device=device)
    total_ctr = torch.tensor(0.0, device=device)
    total_mask = torch.tensor(0.0, device=device)

    # Classification + bbox + centerness losses per FPN level
    for level_idx, (level_out, level_tgt) in enumerate(
        zip(model_output["levels"], level_targets)
    ):
        # Pad cls logits to include background class (class 0)
        cls_logits = level_out["cls"]  # [B, num_cls, H, W]
        # Add background channel
        bg = torch.zeros_like(cls_logits[:, :1])
        cls_logits_with_bg = torch.cat([bg, cls_logits], dim=1)  # [B, num_cls+1, H, W]

        total_cls = total_cls + focal_loss(cls_logits_with_bg, level_tgt["cls"])

        pos_mask = level_tgt["cls"] > 0  # [B, H, W]
        total_bbox = total_bbox + giou_loss(
            level_out["bbox"], level_tgt["bbox"], pos_mask
        )
        total_ctr = total_ctr + F.binary_cross_entropy_with_logits(
            level_out["centerness"][:, 0][pos_mask],
            level_tgt["centerness"][:, 0][pos_mask],
        ) if pos_mask.any() else torch.tensor(0.0, device=device)

    # Mask loss: for each GT instance, compute predicted mask from prototypes
    protos = model_output["protos"]  # [B, K, H, W]
    B = protos.shape[0]
    proto_h, proto_w = protos.shape[2:]

    n_mask_instances = 0
    for b in range(B):
        gt_masks = targets[b]["masks"].to(device)
        n_gt = len(gt_masks)
        if n_gt == 0:
            continue

        gt_masks_small = F.interpolate(
            gt_masks.unsqueeze(1), (proto_h, proto_w),
            mode="bilinear", align_corners=False
        ).squeeze(1)  # [N_gt, proto_h, proto_w]

        # Collect average coefficients per GT instance across all levels
        for g in range(n_gt):
            coeff_sum = torch.zeros(protos.shape[1], device=device)
            coeff_count = 0
            for level_idx, lt in enumerate(level_targets):
                level_coeff = model_output["levels"][level_idx]["coeff"][b]  # [K, H, W]
                assigned = (lt["gt_idx"][b] == g)  # [H, W]
                if assigned.any():
                    # Average coefficients from assigned positions
                    coeff_sum += level_coeff[:, assigned].mean(dim=1)
                    coeff_count += 1

            if coeff_count == 0:
                continue

            avg_coeff = coeff_sum / coeff_count  # [K]
            # Predicted mask = linear combination of prototypes
            pred_mask = (avg_coeff[:, None, None] * protos[b]).sum(dim=0)  # [H, W]

            total_mask = total_mask + F.binary_cross_entropy_with_logits(
                pred_mask, gt_masks_small[g], reduction="mean"
            )
            n_mask_instances += 1

    total_mask = total_mask / max(1, n_mask_instances)

    total = (cfg.cls_weight * total_cls +
             cfg.bbox_weight * total_bbox +
             cfg.bbox_weight * total_ctr +
             cfg.mask_weight * total_mask)

    return total, {
        "cls": total_cls.item(),
        "bbox": total_bbox.item(),
        "ctr": total_ctr.item(),
        "mask": total_mask.item(),
        "total": total.item(),
        "n_mask": n_mask_instances,
    }


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

def train(data_cfg=None, model_cfg=None, train_cfg=None, resume=None):
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Path(model_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = HospitalCOCODataset(
        data_cfg.data_root, split="train", image_size=data_cfg.image_size, augment=True
    )
    val_dataset = HospitalCOCODataset(
        data_cfg.data_root, split="valid", image_size=data_cfg.image_size, augment=False
    )
    num_classes = train_dataset.num_classes  # no +1, background is class 0 in FCOS
    model_cfg.num_classes = num_classes
    print(f"Num classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg.batch_size, shuffle=False,
        num_workers=train_cfg.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Model
    model = ConvNeXtSegModel.build(
        backbone_name=model_cfg.backbone,
        num_classes=num_classes,
        d_model=model_cfg.d_model,
        num_protos=model_cfg.num_prototypes,
        freeze_backbone=(train_cfg.backbone_lr_factor == 0),
    )
    model.to(device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg.learning_rate,
                                  weight_decay=train_cfg.weight_decay)

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
        ckpt = torch.load(os.path.join(resume, "training_state.pt"),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Training
    global_step = 0
    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()
        model.backbone.model.eval()
        epoch_losses = {"cls": 0, "bbox": 0, "ctr": 0, "mask": 0, "total": 0, "n_mask": 0}
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)

            with get_amp_ctx():
                outputs = model(images)

                # Get FPN spatial shapes
                fpn_shapes = [(o["cls"].shape[2], o["cls"].shape[3]) for o in outputs["levels"]]

                # FCOS target assignment
                level_targets = fcos_targets(
                    targets, fpn_shapes, model.strides,
                    data_cfg.image_size, num_classes, device,
                )

                loss, loss_dict = compute_losses(
                    outputs, targets, level_targets, train_cfg, num_classes,
                )
                loss = loss / train_cfg.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % train_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            pbar.set_postfix(
                total=f"{loss_dict['total']:.3f}",
                cls=f"{loss_dict['cls']:.3f}",
                mask=f"{loss_dict['mask']:.3f}",
            )

        n = len(train_loader)
        avg = {k: v / n for k, v in epoch_losses.items()}
        print(f"\n  Epoch {epoch+1} — "
              f"total={avg['total']:.4f}  cls={avg['cls']:.4f}  "
              f"bbox={avg['bbox']:.4f}  mask={avg['mask']:.4f}")

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(device)
                with get_amp_ctx():
                    outputs = model(images)
                    fpn_shapes = [(o["cls"].shape[2], o["cls"].shape[3]) for o in outputs["levels"]]
                    level_targets = fcos_targets(
                        targets, fpn_shapes, model.strides,
                        data_cfg.image_size, num_classes, device,
                    )
                    loss, _ = compute_losses(outputs, targets, level_targets, train_cfg, num_classes)
                val_total += loss.item()

        avg_val = val_total / max(1, len(val_loader))
        print(f"  Epoch {epoch+1} val loss: {avg_val:.4f}")

        # Checkpoint
        ckpt_data = {
            "model_state": model.state_dict(),
            "num_classes": num_classes,
            "backbone": model_cfg.backbone,
            "d_model": model_cfg.d_model,
            "num_prototypes": model_cfg.num_prototypes,
            "image_size": data_cfg.image_size,
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
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train_cfg = TrainConfig()
    data_cfg = DataConfig()
    model_cfg = ModelConfig()

    if args.epochs:      train_cfg.num_epochs = args.epochs
    if args.batch_size:  train_cfg.batch_size = args.batch_size
    if args.lr:          train_cfg.learning_rate = args.lr

    train(data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, resume=args.resume)
