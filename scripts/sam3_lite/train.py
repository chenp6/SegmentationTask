from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from .dataset import COCOSam3LiteTextDataset, collate_fn


DEFAULT_DATA_ROOT = "data/ward_dataset_split/content/dataset_split"


def set_seed(seed: int) -> None:
    # Keep runs reproducible across python/numpy/torch.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss_from_logits(pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Compute soft Dice directly from logits for numerical stability.
    pred = pred_logits.sigmoid()
    inter = (pred * target).sum(dim=(-1, -2))
    union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    return 1.0 - (2.0 * inter + eps) / (union + eps)


def compute_batch_loss(outputs, gt_masks: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    # Model predicts multiple candidate masks per sample; pick the best query per image.
    pred_masks = outputs.pred_masks  # [B, Q, H, W]
    bsz, num_queries, h, w = pred_masks.shape

    gt = F.interpolate(gt_masks.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)  # [B, H, W]
    gt_q = gt.unsqueeze(1).expand(bsz, num_queries, h, w)

    bce_per_q = F.binary_cross_entropy_with_logits(pred_masks, gt_q, reduction="none").mean(dim=(-1, -2))
    dice_per_q = dice_loss_from_logits(pred_masks, gt_q)
    mask_cost = bce_per_q + dice_per_q

    best_idx = torch.argmin(mask_cost, dim=1)  # best query id per image
    mask_loss = mask_cost[torch.arange(bsz, device=pred_masks.device), best_idx].mean()

    score_loss = torch.tensor(0.0, device=pred_masks.device)
    if getattr(outputs, "pred_logits", None) is not None:
        pred_logits = outputs.pred_logits
        pos_scores = pred_logits[torch.arange(bsz, device=pred_logits.device), best_idx]
        score_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))

    presence_loss = torch.tensor(0.0, device=pred_masks.device)
    if getattr(outputs, "presence_logits", None) is not None:
        presence = outputs.presence_logits.view(-1)
        presence_loss = F.binary_cross_entropy_with_logits(presence, torch.ones_like(presence))

    total = mask_loss + 0.25 * score_loss + 0.25 * presence_loss
    logs = {
        "loss_total": float(total.detach().item()),
        "loss_mask": float(mask_loss.detach().item()),
        "loss_score": float(score_loss.detach().item()),
        "loss_presence": float(presence_loss.detach().item()),
    }
    return total, logs


def apply_train_transforms(images, masks, *, hflip_p: float = 0.5, vflip_p: float = 0.1):
    aug_images = []
    aug_masks = []
    for image, mask in zip(images, masks):
        if random.random() < hflip_p:
            image = TF.hflip(image)
            mask = np.flip(mask, axis=1).copy()

        if random.random() < vflip_p:
            image = TF.vflip(image)
            mask = np.flip(mask, axis=0).copy()

        # Color jitter on image only (mask semantics unchanged).
        if random.random() < 0.8:
            b = random.uniform(0.9, 1.1)
            c = random.uniform(0.9, 1.1)
            s = random.uniform(0.9, 1.1)
            image = TF.adjust_brightness(image, b)
            image = TF.adjust_contrast(image, c)
            image = TF.adjust_saturation(image, s)

        aug_images.append(image)
        aug_masks.append(mask)
    return aug_images, aug_masks


def run_epoch(
    *,
    model,
    processor,
    loader,
    device: torch.device,
    optimizer,
    scaler,
    train: bool,
    grad_accum_steps: int,
) -> dict[str, float]:
    # One unified loop for train/valid; toggled by `train`.
    model.train() if train else model.eval()

    running = {"loss_total": 0.0, "loss_mask": 0.0, "loss_score": 0.0, "loss_presence": 0.0}
    steps = 0

    pbar = tqdm(loader, desc="train" if train else "valid")
    optimizer_zeroed = False

    for batch_idx, batch in enumerate(pbar):
        with torch.set_grad_enabled(train):
            images = batch["images"]
            masks = batch["masks"]
            if train:
                images, masks = apply_train_transforms(images, masks)

            inputs = processor(images=images, text=batch["texts"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            gt_masks = torch.from_numpy(np.stack(masks)).float().to(device)

            if train and device.type == "cuda":
                # Mixed precision on CUDA for speed/memory.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)
                    loss, logs = compute_batch_loss(outputs, gt_masks)
                    loss = loss / grad_accum_steps
            else:
                outputs = model(**inputs)
                loss, logs = compute_batch_loss(outputs, gt_masks)
                if train:
                    loss = loss / grad_accum_steps

            if train:
                if not optimizer_zeroed:
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_zeroed = True

                # Support both AMP scaler and regular FP32 backward.
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation step boundary.
                if (batch_idx + 1) % grad_accum_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        for k in running:
            running[k] += logs[k]
        steps += 1
        pbar.set_postfix({"loss": f"{logs['loss_total']:.4f}"})

    # Flush remaining grads when epoch length is not divisible by accum steps.
    if train and steps > 0 and (len(loader) % grad_accum_steps != 0):
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if steps == 0:
        return {k: 0.0 for k in running}
    return {k: v / steps for k, v in running.items()}


def freeze_vision_encoder(model) -> int:
    # Optional transfer-learning mode: freeze vision backbone, train text/head parts.
    frozen = 0
    vision = getattr(model, "vision_model", None) or getattr(model, "vision_encoder", None)
    if vision is None:
        return frozen
    for p in vision.parameters():
        p.requires_grad = False
        frozen += p.numel()
    return frozen


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SAM3-LiteText on COCO dataset")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-name", type=str, default="yonigozlan/sam3-litetext-s0")
    parser.add_argument("--output-dir", type=str, default="output/sam3_lite")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--valid-split", type=str, default="valid")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-vision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")

    train_ds = COCOSam3LiteTextDataset(args.data_root, args.train_split)
    val_ds = COCOSam3LiteTextDataset(args.data_root, args.valid_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    model = AutoModel.from_pretrained(args.model_name)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model.to(device)

    if args.freeze_vision:
        frozen = freeze_vision_encoder(model)
        print(f"Frozen vision params: {frozen:,}")

    total_params, trainable_params = count_params(model)
    print(f"Params total={total_params:,}, trainable={trainable_params:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val = math.inf
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_log = run_epoch(
            model=model,
            processor=processor,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            train=True,
            grad_accum_steps=args.grad_accum_steps,
        )
        with torch.no_grad():
            val_log = run_epoch(
                model=model,
                processor=processor,
                loader=val_loader,
                device=device,
                optimizer=optimizer,
                scaler=None,
                train=False,
                grad_accum_steps=1,
            )

        print(f"  train loss: {train_log['loss_total']:.4f} | valid loss: {val_log['loss_total']:.4f}")
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_log["loss_total"],
                "valid_loss": val_log["loss_total"],
            }
        )

        last_dir = out_dir / "last"
        last_dir.mkdir(parents=True, exist_ok=True)
        # Always save the latest checkpoint.
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        if val_log["loss_total"] < best_val:
            best_val = val_log["loss_total"]
            best_dir = out_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            # Keep a separate best-by-validation-loss checkpoint.
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  Saved best checkpoint to: {best_dir}")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\nTraining finished")
    print(f"Best valid loss: {best_val:.4f}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
