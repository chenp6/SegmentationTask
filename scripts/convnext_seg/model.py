"""
ConvNeXt + YOLACT-style instance segmentation model.

Architecture:
    ConvNeXt-Small (frozen) → FPN → Detection Head + ProtoNet
    Detection: anchor-free FCOS-style (class + bbox + mask coefficients per pixel)
    Masks: linear combination of K prototypes weighted by per-instance coefficients
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Backbone                                                                     #
# --------------------------------------------------------------------------- #

class ConvNeXtBackbone(nn.Module):
    """Frozen ConvNeXt via timm with multi-scale feature extraction."""

    def __init__(self, model_name: str = "convnext_small", freeze: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, features_only=True, out_indices=(1, 2, 3)
        )
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        self.freeze = freeze

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of [C3, C4, C5] feature maps at decreasing resolution."""
        if self.freeze:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)


# --------------------------------------------------------------------------- #
# FPN Neck                                                                     #
# --------------------------------------------------------------------------- #

class FPN(nn.Module):
    """Simple Feature Pyramid Network — top-down + lateral connections."""

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args: features = [C3, C4, C5] from backbone
        Returns: [P3, P4, P5] FPN features (same channels, multi-scale)
        """
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=(h, w), mode="bilinear", align_corners=False
            )

        # 3x3 conv to remove aliasing
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        return outputs


# --------------------------------------------------------------------------- #
# ProtoNet                                                                     #
# --------------------------------------------------------------------------- #

class ProtoNet(nn.Module):
    """Generate K prototype masks from FPN P3 features."""

    def __init__(self, in_channels: int = 256, num_protos: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_protos, 1),  # [B, K, H, W]
        )

    def forward(self, p3: torch.Tensor) -> torch.Tensor:
        """Returns [B, K, H, W] prototype masks at P3 resolution."""
        return self.net(p3)


# --------------------------------------------------------------------------- #
# Detection Head (FCOS-style, anchor-free)                                     #
# --------------------------------------------------------------------------- #

class DetectionHead(nn.Module):
    """
    Shared detection head applied to each FPN level.

    Per-pixel predictions:
        - class: [N_cls] logits (focal loss)
        - bbox:  [4] distances to bbox edges (l, t, r, b)
        - coeff: [K] mask prototype coefficients
        - centerness: [1] scalar for weighting
    """

    def __init__(self, in_channels: int, num_classes: int, num_protos: int = 32):
        super().__init__()
        # Shared classification tower
        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(True),
        )
        self.cls_logits = nn.Conv2d(256, num_classes, 3, padding=1)

        # Shared regression tower
        self.reg_tower = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(True),
        )
        self.bbox_pred = nn.Conv2d(256, 4, 3, padding=1)
        self.centerness = nn.Conv2d(256, 1, 3, padding=1)
        self.coeff_pred = nn.Conv2d(256, num_protos, 3, padding=1)

        # Init classification bias (focal loss stability)
        nn.init.constant_(self.cls_logits.bias, -math.log(99))

    def forward(self, feature: torch.Tensor):
        """
        Args: feature [B, C, H, W] from one FPN level
        Returns dict with cls, bbox, centerness, coeff — all [B, ?, H, W]
        """
        cls_feat = self.cls_tower(feature)
        reg_feat = self.reg_tower(feature)
        return {
            "cls": self.cls_logits(cls_feat),           # [B, num_cls, H, W]
            "bbox": F.relu(self.bbox_pred(reg_feat)),   # [B, 4, H, W] (l,t,r,b distances)
            "centerness": self.centerness(reg_feat),     # [B, 1, H, W]
            "coeff": self.coeff_pred(reg_feat),          # [B, K, H, W]
        }


# --------------------------------------------------------------------------- #
# Full Model                                                                   #
# --------------------------------------------------------------------------- #

class ConvNeXtSegModel(nn.Module):
    """ConvNeXt + FPN + FCOS detection + YOLACT prototype masks."""

    def __init__(
        self,
        backbone: ConvNeXtBackbone,
        fpn: FPN,
        head: DetectionHead,
        protonet: ProtoNet,
        num_classes: int,
        num_protos: int = 32,
    ):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.protonet = protonet
        self.num_classes = num_classes
        self.num_protos = num_protos

        # FPN level strides (relative to input image)
        self.strides = [8, 16, 32]  # for P3, P4, P5 from ConvNeXt

    def forward(self, images: torch.Tensor):
        """
        Returns:
            protos: [B, K, H_p3, W_p3] prototype masks
            level_outputs: list of dicts per FPN level, each with
                cls [B, C, H, W], bbox [B, 4, H, W], centerness [B, 1, H, W], coeff [B, K, H, W]
        """
        features = self.backbone(images)   # [C3, C4, C5]
        fpn_feats = self.fpn(features)     # [P3, P4, P5]

        protos = self.protonet(fpn_feats[0])  # prototypes from P3 (highest res)

        level_outputs = []
        for feat in fpn_feats:
            level_outputs.append(self.head(feat))

        return {"protos": protos, "levels": level_outputs}

    @staticmethod
    def build(
        backbone_name: str = "convnext_small",
        num_classes: int = 27,
        d_model: int = 256,
        num_protos: int = 32,
        freeze_backbone: bool = True,
        **kwargs,
    ) -> "ConvNeXtSegModel":
        print(f"Building ConvNeXt segmentation model...")
        print(f"  Backbone: {backbone_name}")
        print(f"  Classes: {num_classes}, Prototypes: {num_protos}")

        backbone = ConvNeXtBackbone(backbone_name, freeze=freeze_backbone)

        # Probe feature channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            feats = backbone(dummy)
            in_channels = [f.shape[1] for f in feats]
            for i, f in enumerate(feats):
                print(f"    C{i+3}: {f.shape[1]}ch @ {f.shape[2]}×{f.shape[3]}")

        fpn = FPN(in_channels, d_model)
        head = DetectionHead(d_model, num_classes, num_protos)
        protonet = ProtoNet(d_model, num_protos)

        model = ConvNeXtSegModel(backbone, fpn, head, protonet, num_classes, num_protos)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"  Total: {total:,} params ({trainable:,} trainable, {frozen:,} frozen)")

        return model
