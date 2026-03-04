"""
Top-level segmentation model: SAM2 backbone + adapters + decoder.

Supports two decoder types:
  - "query": Mask2Former-lite query-based decoder (recommended)
  - "unet":  UNet with center heatmap + offset heads (legacy)

Usage:
    model = SAM2SegModel.build(num_classes=27, decoder_type="query")
    outputs = model(images)  # {"pred_logits": [B,N,C+1], "pred_masks": [B,N,H,W]}
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .sam2_backbone import SAM2Backbone
from .adapters import MultiStageAdapters


class SAM2SegModel(nn.Module):
    """SAM2 backbone + optional adapters + decoder."""

    def __init__(self, backbone, decoder, adapters=None):
        super().__init__()
        self.backbone = backbone
        self.adapters = adapters
        self.decoder = decoder

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        if self.adapters is not None:
            features = self.adapters(features)
        return self.decoder(features)

    def get_param_groups(self, adapter_lr: float, decoder_lr: float) -> list:
        groups = []
        if self.adapters is not None:
            adapter_params = list(self.adapters.parameters())
            if adapter_params:
                groups.append({"params": adapter_params, "lr": adapter_lr})
        decoder_params = list(self.decoder.parameters())
        if decoder_params:
            groups.append({"params": decoder_params, "lr": decoder_lr})
        return groups

    @classmethod
    def build(
        cls,
        model_name: str = "facebook/sam2.1-hiera-large",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        num_classes: int = 27,
        use_adapters: bool = True,
        adapter_dim: int = 64,
        image_size: int = 1024,
        decoder_type: str = "query",
        num_queries: int = 100,
        decoder_layers: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 1024,
    ) -> "SAM2SegModel":
        print(f"Building SAM2 segmentation model...")
        print(f"  Model: {model_name}")
        print(f"  Classes: {num_classes}, Decoder: {decoder_type}")
        print(f"  Adapters: {use_adapters} (dim={adapter_dim})")

        backbone = SAM2Backbone(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            freeze=True,
        )

        print(f"  Probing backbone feature shapes...")
        feat_info = backbone.get_feature_info(image_size)
        for name, (c, h, w) in feat_info.items():
            print(f"    {name}: C={c}, H={h}, W={w}")

        adapters = None
        if use_adapters:
            stage_channels = {name: shape[0] for name, shape in feat_info.items()}
            adapters = MultiStageAdapters(stage_channels, adapter_dim)
            print(f"  Adapter params: {sum(p.numel() for p in adapters.parameters()):,}")

        stage_specs = [(name, s[0], s[1], s[2]) for name, s in feat_info.items()]

        if decoder_type == "query":
            from .query_decoder import QueryInstanceDecoder
            decoder = QueryInstanceDecoder(
                stage_channels=stage_specs,
                num_classes=num_classes,
                d_model=d_model,
                num_queries=num_queries,
                num_layers=decoder_layers,
                nhead=nhead,
                dim_ff=dim_ff,
            )
        else:
            from .unet_decoder import UNetDecoder
            decoder = UNetDecoder(
                stage_channels=stage_specs,
                num_classes=num_classes,
                input_size=image_size,
            )

        print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

        model = cls(backbone=backbone, decoder=decoder, adapters=adapters)
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_all:,} ({total_trainable:,} trainable)")
        return model
