"""
UNet-style decoder consuming multi-scale features from the SAM3 backbone.

Architecture (bottom-up):
  For each level from deepest to shallowest:
    1. Bilinear upsample the lower-resolution feature
    2. Concatenate with the skip connection from the encoder
    3. Apply 2× (Conv3×3 + BN + ReLU)
  Final: 1×1 conv to map to num_classes, then upsample to input resolution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class DoubleConv(nn.Module):
    """Two consecutive (Conv3×3 + BN + ReLU) blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    """
    UNet decoder that takes multi-scale features and produces segmentation logits.

    Args:
        stage_channels: list of (stage_name, C, H, W) from shallowest to deepest
                        e.g. [("stage0", 256, 64, 64), ("stage1", 256, 32, 32),
                              ("stage2", 256, 16, 16)]
        num_classes: number of output segmentation classes (including background)
        decoder_channels: channel dimensions for each decoder level
                         (default: automatically computed)
        input_size: original input image size for final upsampling
    """

    def __init__(
        self,
        stage_channels: List[Tuple[str, int, int, int]],
        num_classes: int,
        decoder_channels: List[int] = None,
        input_size: int = 1024,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Sort stages from deepest (smallest spatial) to shallowest (largest)
        self.stages = sorted(stage_channels, key=lambda x: x[2])  # sort by H
        self.stage_names = [s[0] for s in self.stages]

        n_stages = len(self.stages)
        if decoder_channels is None:
            decoder_channels = [256] * (n_stages - 1)

        # Build decoder levels (from deep to shallow)
        self.up_convs = nn.ModuleList()
        self.double_convs = nn.ModuleList()

        in_ch = self.stages[0][1]  # deepest stage channels

        for i in range(1, n_stages):
            skip_ch = self.stages[i][1]  # skip connection channels
            dec_ch = decoder_channels[i - 1] if i - 1 < len(decoder_channels) else 256

            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
            )
            self.double_convs.append(
                DoubleConv(in_ch + skip_ch, dec_ch)
            )
            in_ch = dec_ch

        # Segmentation head
        self.seg_head = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: dict of stage_name -> tensor [B, C, H, W]

        Returns:
            logits: [B, num_classes, H_in, W_in]
        """
        # Collect features ordered from deepest to shallowest
        ordered_feats = [features[name] for name in self.stage_names]

        x = ordered_feats[0]  # start from deepest
        for i in range(len(self.up_convs)):
            skip = ordered_feats[i + 1]

            x = self.up_convs[i](x)

            # Match spatial dimensions (in case of slight mismatch)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = self.double_convs[i](x)

        # Segmentation head
        logits = self.seg_head(x)

        # Upsample to input resolution
        if logits.shape[2:] != (self.input_size, self.input_size):
            logits = F.interpolate(
                logits,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        return logits
