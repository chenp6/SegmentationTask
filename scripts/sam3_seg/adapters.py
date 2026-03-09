"""
Lightweight adapter modules for parameter-efficient finetuning of SAM3 backbone.

Each adapter is a residual bottleneck applied to a frozen encoder stage output:
    Adapter(x) = x + W_up(ReLU(W_down(x)))

Using 1×1 convolutions for spatial feature maps.
"""
import torch
import torch.nn as nn
from typing import Dict, List


class ConvAdapter(nn.Module):
    """
    1×1 conv bottleneck adapter for a single feature stage.

    Args:
        in_channels: number of input channels (from backbone stage)
        adapter_dim: bottleneck dimension (default 64)
    """

    def __init__(self, in_channels: int, adapter_dim: int = 64):
        super().__init__()
        self.down = nn.Conv2d(in_channels, adapter_dim, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(adapter_dim, in_channels, kernel_size=1, bias=False)

        # Initialize near-zero so adapter starts as identity
        nn.init.kaiming_normal_(self.down.weight, mode="fan_out")
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class MultiStageAdapters(nn.Module):
    """
    Collection of ConvAdapters, one per backbone stage.

    Args:
        stage_channels: dict mapping stage_name -> channel count
        adapter_dim: bottleneck dimension for all adapters
    """

    def __init__(self, stage_channels: Dict[str, int], adapter_dim: int = 64):
        super().__init__()
        self.adapters = nn.ModuleDict({
            name: ConvAdapter(channels, adapter_dim)
            for name, channels in stage_channels.items()
        })

    def forward(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply per-stage adapters to backbone features."""
        return {
            name: self.adapters[name](feat) if name in self.adapters else feat
            for name, feat in features.items()
        }
