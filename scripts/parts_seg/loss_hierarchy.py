"""
Segmentation loss + auxiliary object (hierarchy) CE at positive anchors.

Expects batch to have "object_id" (n_instances,) and preds to have "object_scores" (B, no_object, n_anchors).
Uses the same positive assignment as v8 (fg_mask, target_gt_idx) to get object_id per positive anchor.
"""
import torch
import torch.nn.functional as F


def _global_gt_indices(batch_idx: torch.Tensor, target_gt_idx: torch.Tensor, fg_mask: torch.Tensor, batch_size: int):
    """Map (batch_idx, target_gt_idx) at positive anchors to global instance indices.

    batch_idx: (n_instances,) which image each instance belongs to.
    target_gt_idx: (B, n_anchors) index of assigned GT in each image.
    fg_mask: (B, n_anchors) bool.
    Returns: (n_positives,) global indices into the instance list.
    """
    device = fg_mask.device
    n_per_image = [(batch_idx == b).sum().item() for b in range(batch_size)]
    offset = torch.tensor([0] + [sum(n_per_image[:b]) for b in range(1, batch_size)], device=device, dtype=torch.long)
    # Positive positions: (batch_idx_anchors, gt_idx_in_image)
    batch_idx_anchors = torch.arange(batch_size, device=device).view(-1, 1).expand_as(target_gt_idx)
    flat_b = batch_idx_anchors.flatten()[fg_mask.flatten()]
    flat_gt = target_gt_idx.flatten().long()[fg_mask.flatten()]
    global_idx = offset[flat_b] + flat_gt
    return global_idx


class v8SegmentationLossWithObject:
    """Wraps v8SegmentationLoss and adds object auxiliary CE at positive anchors."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None, object_loss_gain: float = 0.5):
        from ultralytics.utils.loss import v8SegmentationLoss

        self._seg_loss = v8SegmentationLoss(model, tal_topk, tal_topk2)
        self.object_loss_gain = object_loss_gain
        self.device = self._seg_loss.device

    def __call__(self, preds, batch):
        preds_parsed = self._seg_loss.parse_output(preds)
        return self.loss(preds_parsed, batch)

    def loss(self, preds: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # Part (box, seg, cls, dfl) + optional object. Always return 6 loss_items so validator matches trainer.
        total, loss_detach = self._seg_loss.loss(preds, batch)
        batch_size = preds["boxes"].shape[0]

        def add_object_slot(detach, value=None):
            v = value if value is not None else torch.zeros(1, device=detach.device, dtype=detach.dtype)
            return torch.cat([detach, v.unsqueeze(0) if v.dim() == 0 else v.view(1)])

        if "object_scores" not in preds or "object_id" not in batch:
            return total, add_object_slot(loss_detach, None)

        object_scores = preds["object_scores"]  # (B, no_object, n_anchors)
        object_id = batch["object_id"].to(self.device).long()  # (n_instances,)
        (fg_mask, target_gt_idx, _, _, _), _, _ = self._seg_loss.get_assigned_targets_and_loss(preds, batch)
        n_pos = fg_mask.sum().item()
        if n_pos == 0:
            return total, add_object_slot(loss_detach, None)

        global_gt_idx = _global_gt_indices(
            batch["batch_idx"].view(-1), target_gt_idx, fg_mask, batch_size
        )
        if global_gt_idx.numel() == 0:
            return total, add_object_slot(loss_detach, None)

        no_object = object_scores.shape[1]
        n_instances = object_id.shape[0]
        # Bounds checks to avoid CUDA device-side assert (OOB or invalid class index)
        valid_idx = (global_gt_idx >= 0) & (global_gt_idx < n_instances)
        global_gt_idx = global_gt_idx[valid_idx]
        if global_gt_idx.numel() == 0:
            return total, add_object_slot(loss_detach, None)

        # Predicted logits at positive anchors: (n_pos, no_object)
        flat_scores = object_scores.permute(0, 2, 1).contiguous().view(-1, no_object)
        positive_anchor_flat = fg_mask.flatten()
        pred_object = flat_scores[positive_anchor_flat][valid_idx]  # (n_valid, no_object)
        target_object = object_id[global_gt_idx]  # (n_valid,)
        valid_obj = (target_object >= 0) & (target_object < no_object)
        if valid_obj.sum() == 0:
            return total, add_object_slot(loss_detach, None)
        pred_object = pred_object[valid_obj]
        target_object = target_object[valid_obj]
        object_loss = F.cross_entropy(pred_object, target_object, reduction="mean")
        object_loss = object_loss.clamp(max=10.0)
        total = total + self.object_loss_gain * object_loss * batch_size
        loss_detach = add_object_slot(loss_detach, object_loss.detach())
        return total, loss_detach
