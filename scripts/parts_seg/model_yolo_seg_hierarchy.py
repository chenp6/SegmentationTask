"""
YOLOv11-seg backbone (not YOLO26) + custom head with 234 part classes and 44 object (auxiliary) classes.

Uses yolo11n-seg.pt / yolo11s-seg.pt etc. Builds a SegmentationModel with nc=234, replaces the Segment
head with SegmentWithObject so we keep box/mask/proto and add object_scores for hierarchy auxiliary loss.
"""
from pathlib import Path
import copy

import torch
import torch.nn as nn

# Default neck output channels for YOLOv11n-seg (P3, P4, P5); actual ch read from checkpoint
YOLO11_SEG_CH = (256, 512, 1024)


class SegmentWithObject(nn.Module):
    """Segment head with an auxiliary object-class head (for hierarchy loss).

    Same as Ultralytics Segment but:
    - nc = part classes (e.g. 234)
    - Adds cv_object: outputs no_object (e.g. 44) per anchor
    - forward() adds preds['object_scores'] in training.
    """

    def __init__(
        self,
        nc: int,
        no_object: int = 44,
        nm: int = 32,
        npr: int = 256,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = YOLO11_SEG_CH,
    ):
        super().__init__()
        from ultralytics.nn.modules.head import Segment
        from ultralytics.nn.modules.conv import Conv

        self._base = Segment(nc=nc, nm=nm, npr=npr, reg_max=reg_max, end2end=end2end, ch=ch)
        self.nc = nc
        self.reg_max = reg_max  # required by Ultralytics loss (reads from model.model[-1])
        self.no_object = no_object
        self.nl = self._base.nl
        self.nm = self._base.nm
        self.stride = self._base.stride
        self.ch = ch
        # Auxiliary head: one conv per level -> no_object logits
        self.cv_object = nn.ModuleList(nn.Conv2d(ch[i], no_object, 1) for i in range(self.nl))

    def forward(self, x):
        preds = self._base(x)
        if self.training and isinstance(preds, dict):
            bs = x[0].shape[0]
            object_scores = torch.cat(
                [self.cv_object[i](x[i]).view(bs, self.no_object, -1) for i in range(self.nl)], dim=-1
            )
            if "one2many" in preds and "one2one" in preds:
                preds["one2many"]["object_scores"] = object_scores
                preds["one2one"]["object_scores"] = object_scores.detach()
            else:
                preds["object_scores"] = object_scores
        return preds

    def bias_init(self):
        self._base.bias_init()

    @property
    def end2end(self):
        return getattr(self._base, "end2end", True)

    def fuse(self):
        self._base.fuse()


def build_yolo_seg_hierarchy_model(
    checkpoint: str = "yolo11n-seg.pt",
    nc_part: int = 234,
    no_object: int = 44,
    device: str | None = None,
):
    """Load YOLOv11-seg and replace the head with SegmentWithObject (234 part + 44 object).

    Backbone and neck weights are loaded from checkpoint; the class head is re-initialized
    for nc_part, and the object head is new. Box/mask/proto are copied from the pretrained head.
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel

    # Trainer may pass a checkpoint dict; we need a path string for YOLO()
    if not isinstance(checkpoint, (str, Path)) or len(str(checkpoint)) > 260:
        checkpoint = "yolo11n-seg.pt"
    checkpoint = str(checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(checkpoint)
    inner = model.model
    if not isinstance(inner, SegmentationModel):
        raise ValueError(f"Expected SegmentationModel (YOLOv11-seg), got {type(inner)}")

    old_head = inner.model[-1]
    # Infer input channels from the old head's state_dict (box head first conv per level)
    old_state = old_head.state_dict()
    try:
        ch = tuple(
            int(old_state[f"cv2.{i}.0.conv.weight"].shape[1])
            for i in range(3)
        )
    except (KeyError, IndexError):
        ch = YOLO11_SEG_CH

    new_head = SegmentWithObject(
        nc=nc_part,
        no_object=no_object,
        nm=getattr(old_head, "nm", 32),
        npr=getattr(old_head, "npr", 256),
        reg_max=getattr(old_head, "reg_max", 16),
        end2end=getattr(old_head, "end2end", True),
        ch=ch,
    )
    new_head = new_head.to(device)
    if hasattr(old_head, "stride") and old_head.stride is not None and old_head.stride.numel():
        new_head.stride = old_head.stride.to(device)
        new_head._base.stride = new_head.stride

    # Copy box/mask/proto/dfl and one2one box/mask from pretrained into _base (skip cv3 - different nc)
    old_state = old_head.state_dict()
    new_state = new_head.state_dict()
    copy_prefixes = ("cv2.", "cv4.", "proto.", "dfl.", "one2one_cv2.", "one2one_cv4.")
    for old_k, old_v in old_state.items():
        if any(old_k.startswith(p) for p in copy_prefixes):
            new_k = "_base." + old_k
            if new_k in new_state and new_state[new_k].shape == old_v.shape:
                new_state[new_k] = old_v.clone()
    new_head.load_state_dict(new_state, strict=False)
    try:
        new_head.bias_init()  # reinit cls bias for nc_part (needs stride)
    except (ValueError, ZeroDivisionError):
        pass  # stride may be unset until first forward

    # Preserve layer routing attributes used by DetectionModel.predict()
    for attr in ("f", "i", "type"):
        if hasattr(old_head, attr):
            setattr(new_head, attr, getattr(old_head, attr))
    inner.model[-1] = new_head
    inner.nc = nc_part
    return model


def load_yolo_seg_hierarchy_from_checkpoint(
    ckpt_path: str | Path,
    data_yaml: str | Path | None = None,
    device: str | None = None,
):
    """Load a trained hierarchy model from an Ultralytics training checkpoint (e.g. best.pt).

    The checkpoint stores the EMA model; we build the hierarchy architecture and load
    that state so eval/predict match training.

    Returns:
        (YOLO, ckpt_dict): The YOLO wrapper (use .model for inner, .val() / .predict() for inference).
    """
    from ultralytics.utils import YAML

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        # Try relative to repo root (e.g. when cwd is not repo root)
        repo_root = Path(__file__).resolve().parents[2]
        alt = repo_root / ckpt_path
        if alt.is_file():
            ckpt_path = alt
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path} (also tried {alt}). "
                "Use last.pt if best.pt is not saved yet (best.pt appears when validation improves)."
            )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Checkpoint EMA was saved with init_criterion = MethodType(_hierarchy_init_criterion, model); unpickler needs that name on the class
    from ultralytics.nn.tasks import SegmentationModel
    if not hasattr(SegmentationModel, "_hierarchy_init_criterion"):
        SegmentationModel._hierarchy_init_criterion = lambda model: None  # stub for unpickling; we only use state_dict
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Infer nc and names from data yaml or from training args in checkpoint
    nc_part = 234
    data = None
    if data_yaml:
        path = Path(data_yaml)
        if path.is_file() and str(path).endswith((".yaml", ".yml")):
            data = YAML.load(path)
            nc_part = int(data.get("nc", nc_part)) if data else nc_part
    if data is None and isinstance(ckpt.get("train_args"), dict):
        data_str = ckpt["train_args"].get("data", "")
        if data_str:
            path = Path(data_str)
            if path.is_file() and str(path).endswith((".yaml", ".yml")):
                data = YAML.load(path)
                if data:
                    nc_part = int(data.get("nc", nc_part))

    yolo_model = build_yolo_seg_hierarchy_model(
        checkpoint="yolo11n-seg.pt",
        nc_part=nc_part,
        no_object=44,
        device=device,
    )
    # Load EMA weights (trainer saves ema, not model)
    ema_state = ckpt.get("ema")
    if ema_state is not None:
        state = ema_state.float().state_dict()
    else:
        state = ckpt.get("model")
    if state is None:
        raise ValueError("Checkpoint has no 'ema' or 'model' state dict")
    yolo_model.model.load_state_dict(state, strict=True)
    yolo_model.model.to(device)
    # Validator uses len(model.names) for nc in NMS; base YOLO has 80 names so "extra" (mask coeffs) would be wrong
    names = None
    if data and isinstance(data.get("names"), list) and len(data["names"]) == nc_part:
        names = {i: data["names"][i] for i in range(nc_part)}
    if names is None or len(names) != nc_part:
        names = {i: f"part_{i}" for i in range(nc_part)}
    yolo_model.model.names = names
    return yolo_model, ckpt
