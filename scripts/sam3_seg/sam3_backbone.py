import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SAM3Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/sam3", checkpoint_path: Optional[str] = None, config_path: Optional[str] = None, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self._is_timm_fallback = False
        self._resize_warned = False
        self.encoder = self._load_encoder(model_name, checkpoint_path, config_path)

        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_encoder(self, model_name, checkpoint_path, config_path):
        """Load the SAM3 encoder using official SAM3 API."""
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading SAM3 encoder from local checkpoint: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                model = self._build_sam3_model()
                model.load_state_dict(state_dict)
                encoder = model.image_encoder if hasattr(model, "image_encoder") else model.encoder
                return encoder

            if model_name == "facebook/sam3":
                print(f"Loading SAM3 encoder using official API: {model_name}")
                from sam3.model_builder import build_sam3_image_model

                model = build_sam3_image_model()
                encoder = model.image_encoder if hasattr(model, "image_encoder") else model.encoder
                return encoder

            print(f"Loading generic model: {model_name}")
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
            return model

        except Exception as e:
            print(f"Error loading SAM3 encoder: {e}")
            print("Trying fallback: Vision Transformer")

            try:
                import timm
                self._is_timm_fallback = True
                encoder = timm.create_model("vit_large_patch14_clip_336", pretrained=True)
                return encoder
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise

    def _build_sam3_model(self):
        """Build a basic SAM3 model structure for loading checkpoints."""
        try:
            from sam3.model_builder import build_sam3_image_model
            return build_sam3_image_model()
        except ImportError:
            print("SAM3 package not available. Using placeholder architecture...")
            import timm
            return timm.create_model("vit_large_patch14_clip_336", pretrained=False)

    def _get_input_size(self) -> int:
        """Get the expected input size for the encoder."""
        try:
            if hasattr(self.encoder, "default_cfg"):
                input_size = self.encoder.default_cfg.get("input_size", [3, 224, 224])
                if isinstance(input_size, (list, tuple)) and len(input_size) >= 2:
                    return int(input_size[-1])

            if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "image_size"):
                return int(self.encoder.config.image_size)

            if hasattr(self.encoder, "image_size"):
                return int(self.encoder.image_size)

            if hasattr(self.encoder, "patch_embed") and hasattr(self.encoder.patch_embed, "img_size"):
                img_size = self.encoder.patch_embed.img_size
                if isinstance(img_size, (list, tuple)):
                    return int(img_size[-1])
                return int(img_size)

            return 224

        except Exception as e:
            print(f"Could not determine input size: {e}")
            return 224

    def get_feature_info(self, image_size: Optional[int] = None) -> Dict[str, tuple]:
        """Get feature information by running a dummy forward pass."""
        if image_size is None:
            image_size = self._get_input_size()

        dummy_input = torch.randn(1, 3, image_size, image_size).to(next(self.parameters()).device)

        with torch.no_grad():
            features = self.forward(dummy_input)

        feature_info = {}
        if isinstance(features, dict):
            for stage_name, feat in features.items():
                if isinstance(feat, torch.Tensor) and feat.dim() == 4:
                    c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
                    feature_info[stage_name] = (c, h, w)
        elif isinstance(features, (list, tuple)):
            for i, feat in enumerate(features):
                if isinstance(feat, torch.Tensor) and feat.dim() == 4:
                    c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
                    feature_info[f"stage{i}"] = (c, h, w)
        elif isinstance(features, torch.Tensor) and features.dim() == 4:
            c, h, w = features.shape[1], features.shape[2], features.shape[3]
            feature_info["stage0"] = (c, h, w)

        if not feature_info:
            raise RuntimeError("Backbone did not return 4D feature maps; cannot build segmentation decoder.")

        return feature_info

    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from encoder outputs into {stage: tensor}."""
        if isinstance(x, dict):
            return x

        if isinstance(x, (list, tuple)):
            return {f"stage{i}": feat for i, feat in enumerate(x) if isinstance(feat, torch.Tensor)}

        if isinstance(x, torch.Tensor) and x.dim() == 3:
            # ViT token output [B, N, C] -> [B, C, H, W]
            b, n, c = x.shape
            side = int(math.sqrt(n))
            tokens = x
            if side * side != n and n > 1:
                side = int(math.sqrt(n - 1))
                if side * side == (n - 1):
                    tokens = x[:, 1:, :]  # drop CLS token
                else:
                    raise ValueError(f"Cannot reshape token sequence of length {n} into square feature map.")
            fmap = tokens.transpose(1, 2).reshape(b, c, side, side)
            return {"stage0": fmap}

        return {"stage0": x}

    def _resize_input_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.encoder, "patch_embed"):
            return x

        img_size = getattr(self.encoder.patch_embed, "img_size", None)
        if img_size is None:
            return x

        if isinstance(img_size, (list, tuple)):
            target_h, target_w = int(img_size[0]), int(img_size[1])
        else:
            target_h = target_w = int(img_size)

        if x.shape[-2:] == (target_h, target_w):
            return x

        if not self._resize_warned:
            print(
                f"Input size {x.shape[-2:]} does not match encoder size {(target_h, target_w)}; "
                "resizing input for fallback encoder."
            )
            self._resize_warned = True

        return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the encoder."""
        encoder_input = self._resize_input_if_needed(x)

        if self.freeze:
            with torch.no_grad():
                if self._is_timm_fallback and hasattr(self.encoder, "forward_features"):
                    encoder_output = self.encoder.forward_features(encoder_input)
                else:
                    encoder_output = self.encoder(encoder_input)
        else:
            if self._is_timm_fallback and hasattr(self.encoder, "forward_features"):
                encoder_output = self.encoder.forward_features(encoder_input)
            else:
                encoder_output = self.encoder(encoder_input)

        features = self._extract_features(encoder_output)

        if self.freeze:
            features = {k: v.detach() for k, v in features.items()}

        return features
