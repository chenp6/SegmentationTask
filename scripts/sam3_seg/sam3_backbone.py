"""
SAM3 Hiera image encoder as a multi-scale feature backbone.

Loads the SAM3 model and exposes only the image encoder (Hiera),
returning multi-scale feature maps suitable for a UNet-style decoder.

Usage:
    backbone = SAM3Backbone(model_name="facebook/sam3-hiera-large")
    features = backbone(images)  # images: [B, 3, H, W]
    # features: {"stage0": [B,C0,H0,W0], ..., "stage3": [B,C3,H3,W3]}
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class SAM3Backbone(nn.Module):
    """
    Wraps the SAM3 image encoder (Hiera) to produce multi-scale features.

    The Hiera backbone returns a list of feature maps at different resolutions.
    We expose them as a dict keyed by stage name for the decoder to consume.
    """

    def __init__(
        self,
        model_name: str = "facebook/sam3-hiera-large",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self._freeze = freeze

        # Load the SAM3 image encoder
        self.encoder = self._load_encoder(model_name, checkpoint_path, config_path)

        if freeze:
            self._freeze_encoder()

    def _load_encoder(self, model_name, checkpoint_path, config_path):
        """Load SAM3 image encoder from HuggingFace or local checkpoint."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try local checkpoint + config first
        try:
            from sam3.build_sam import build_sam3
            if checkpoint_path and config_path:
                sam3_model = build_sam3(
                    config_path, checkpoint_path, device=device
                )
                return sam3_model.image_encoder
        except ImportError:
            pass

        # Try loading via sam3 package from HF hub
        try:
            from sam3.build_sam import build_sam3_hf
            sam3_model = build_sam3_hf(model_name, device=device)
            return sam3_model.image_encoder
        except (ImportError, Exception) as e:
            last_error = e

        # Fall back to HuggingFace transformers SAM3
        try:
            from transformers import Sam3Model
            sam3_model = Sam3Model.from_pretrained(model_name)
            sam3_model.to(device)
            return sam3_model.image_encoder
        except (ImportError, Exception) as e:
            last_error = e

        raise RuntimeError(
            f"Could not load SAM3 model '{model_name}'. "
            f"Install sam3: pip install git+https://github.com/facebookresearch/sam3.git\n"
            f"Error: {last_error}"
        )

    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def get_feature_info(self, image_size: int = 1024) -> Dict[str, tuple]:
        """
        Run a dummy forward pass to discover feature map shapes.
        Returns dict: stage_name -> (C, H, W).
        """
        device = next(self.encoder.parameters()).device
        dummy = torch.randn(1, 3, image_size, image_size, device=device)
        features = self._extract_features(dummy)
        return {name: tuple(feat.shape[1:]) for name, feat in features.items()}

    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run image through encoder and collect multi-scale features.

        SAM3's image encoder (Hiera with FpnNeck) returns a dict with:
          - "backbone_fpn": list of feature tensors from FPN levels
          - "vision_pos_enc": list of positional encodings

        We extract just the backbone_fpn features as our multi-scale outputs.
        """
        # The SAM3 image encoder's forward returns different structures
        # depending on version. Handle both cases.
        output = self.encoder(x)

        if isinstance(output, dict):
            # Standard SAM3 output: dict with "backbone_fpn" key
            if "backbone_fpn" in output:
                fpn_features = output["backbone_fpn"]
                return {
                    f"stage{i}": feat for i, feat in enumerate(fpn_features)
                }
            # Some versions return vision features directly
            elif "vision_features" in output:
                return {"stage0": output["vision_features"]}

        if isinstance(output, (list, tuple)):
            return {f"stage{i}": feat for i, feat in enumerate(output)}

        if isinstance(output, torch.Tensor):
            return {"stage0": output}

        raise ValueError(
            f"Unexpected encoder output type: {type(output)}. "
            f"Keys: {output.keys() if hasattr(output, 'keys') else 'N/A'}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: input images [B, 3, H, W], ImageNet-normalised float32

        Returns:
            Dict of multi-scale features:
              "stage0": highest resolution features
              ...
              "stageN": lowest resolution (deepest) features
        """
        if self._freeze:
            with torch.no_grad():
                features = self._extract_features(x)
            # Detach to avoid graph issues but keep grad for downstream
            return {k: v.detach() for k, v in features.items()}
        else:
            return self._extract_features(x)

    @property
    def num_stages(self) -> int:
        """Number of feature stages (discovered on first forward pass)."""
        return len(self.get_feature_info())
