"""
Query-based instance segmentation decoder (Mask2Former-lite).

Learnable queries attend to pixel features via cross-attention and
produce per-instance class predictions + binary masks.

Architecture per layer:
  Self-Attention → Cross-Attention → FFN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class QueryDecoderLayer(nn.Module):
    """Single transformer decoder layer: self-attn → cross-attn → FFN."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention: queries attend to pixel features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, pixel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries:         [B, N, D] learnable queries
            pixel_features:  [B, HW, D] flattened pixel features
        Returns:
            queries: [B, N, D] updated queries
        """
        # Self-attention
        q = self.norm1(queries)
        q2, _ = self.self_attn(q, q, q)
        queries = queries + q2

        # Cross-attention to pixel features
        q = self.norm2(queries)
        q2, _ = self.cross_attn(q, pixel_features, pixel_features)
        queries = queries + q2

        # FFN
        q = self.norm3(queries)
        queries = queries + self.ffn(q)

        return queries


class PixelDecoder(nn.Module):
    """Simplified UNet-style pixel decoder — outputs features only, no task heads.

    Args:
        max_decode_stages: limit how many skip connections to use.
            None = all (256²), 1 = stop at 128², etc.
    """

    def __init__(self, stage_channels: List[Tuple[str, int, int, int]], d_model: int = 256,
                 max_decode_stages: int = None):
        super().__init__()
        self.d_model = d_model

        # Sort stages from deepest (smallest spatial) to shallowest (largest)
        self.stages = sorted(stage_channels, key=lambda x: x[2])
        self.stage_names = [s[0] for s in self.stages]

        n_stages = len(self.stages)
        n_decode = min(n_stages - 1, max_decode_stages) if max_decode_stages else n_stages - 1

        self.up_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()

        in_ch = self.stages[0][1]
        for i in range(1, n_decode + 1):
            skip_ch = self.stages[i][1]
            self.up_convs.append(nn.ConvTranspose2d(in_ch, d_model, kernel_size=2, stride=2))
            self.fuse_convs.append(nn.Sequential(
                nn.Conv2d(d_model + skip_ch, d_model, 3, padding=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.ReLU(inplace=True),
            ))
            in_ch = d_model

        # Project deepest stage if needed
        if self.stages[0][1] != d_model:
            self.input_proj = nn.Conv2d(self.stages[0][1], d_model, 1)
        else:
            self.input_proj = nn.Identity()

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns pixel features [B, D, H, W]."""
        ordered = [features[name] for name in self.stage_names]
        x = self.input_proj(ordered[0])

        for i in range(len(self.up_convs)):
            skip = ordered[i + 1]
            x = self.up_convs[i](x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.fuse_convs[i](x)

        return x  # [B, D, H_dec, W_dec]


class QueryInstanceDecoder(nn.Module):
    """
    Mask2Former-lite: query-based instance segmentation decoder.

    Speed optimization: cross-attention runs at 64² (4K tokens), not 256² (65K).
    Full-resolution pixel features used ONLY for the final mask dot product.
    """

    def __init__(
        self,
        stage_channels: List[Tuple[str, int, int, int]],
        num_classes: int,
        d_model: int = 256,
        num_queries: int = 100,
        num_layers: int = 3,
        nhead: int = 8,
        dim_ff: int = 1024,
        attn_resolution: int = 64,  # cross-attention spatial resolution
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model
        self.attn_resolution = attn_resolution

        # Pixel decoder — stop at 128² (1 skip), not 256² (2 skips)
        # Saves 4× on mask dot product and loss
        self.pixel_decoder = PixelDecoder(stage_channels, d_model, max_decode_stages=1)

        # Downsample projection for cross-attention (128² → 64²)
        self.attn_downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        nn.init.xavier_uniform_(self.query_embed.weight)

        # Sinusoidal positional encoding for attention features
        self.register_buffer("_attn_pe", self._build_sinusoidal_pe(attn_resolution, d_model))

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            QueryDecoderLayer(d_model, nhead, dim_ff) for _ in range(num_layers)
        ])

        # Output heads
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _build_sinusoidal_pe(resolution: int, d_model: int) -> torch.Tensor:
        """Build 2D sinusoidal positional encoding [1, H*W, D]."""
        pe = torch.zeros(resolution * resolution, d_model)
        position = torch.arange(resolution * resolution).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(4.0 / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, HW, D]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Returns:
            "pred_logits": [B, N, num_classes+1]
            "pred_masks":  [B, N, H_dec, W_dec] (full pixel decoder resolution)
        """
        # Full-res pixel features for mask prediction
        pixel_feat = self.pixel_decoder(features)  # [B, D, H, W] ~256²
        B, D, H, W = pixel_feat.shape

        # Downsampled features for cross-attention (64² = 4K tokens, not 65K)
        attn_feat = self.attn_downsample(pixel_feat)  # [B, D, ~64, ~64]
        aH, aW = attn_feat.shape[2:]
        attn_flat = attn_feat.flatten(2).permute(0, 2, 1)  # [B, aH*aW, D]

        # Add positional encoding (resize if needed)
        if self._attn_pe.shape[1] != aH * aW:
            pe = F.interpolate(
                self._attn_pe.permute(0, 2, 1).reshape(1, D, self.attn_resolution, self.attn_resolution),
                size=(aH, aW), mode="bilinear", align_corners=False
            ).flatten(2).permute(0, 2, 1)
        else:
            pe = self._attn_pe
        attn_flat = attn_flat + pe.to(attn_flat.dtype)

        # Init queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Transformer decoder — cross-attention on 4K tokens (fast!)
        for layer in self.layers:
            queries = layer(queries, attn_flat)

        # Class predictions
        pred_logits = self.class_head(queries)  # [B, N, C+1]

        # Mask predictions — dot product with FULL-RES pixel features
        mask_embed = self.mask_head(queries)  # [B, N, D]
        pred_masks = torch.einsum("bnd,bdhw->bnhw", mask_embed, pixel_feat)

        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
        }

