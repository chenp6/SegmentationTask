import os
import torch
import torch.nn as nn
from typing import Optional

class SAM3Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/sam3", checkpoint_path: Optional[str] = None, config_path: Optional[str] = None, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.encoder = self._load_encoder(model_name, checkpoint_path, config_path)
        
        # 如果需要凍結參數
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _load_encoder(self, model_name, checkpoint_path, config_path):
        """Load the SAM3 encoder using official SAM3 API."""
        try:
            # 優先使用本地 checkpoint
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading SAM3 encoder from local checkpoint: {checkpoint_path}")
                # 如果是本地 checkpoint，使用 torch.load
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                # 建立模型並載入權重
                model = self._build_sam3_model()
                model.load_state_dict(state_dict)
                encoder = model.encoder  # 假設模型有 .encoder 屬性
                return encoder
            
            # 使用官方 SAM3 API 載入
            if model_name == "facebook/sam3":
                print(f"Loading SAM3 encoder using official API: {model_name}")
                from sam3.model_builder import build_sam3_image_model
                
                # 載入完整的 SAM3 模型
                model = build_sam3_image_model()
                
                # 提取 encoder 部分（根據 SAM3 架構調整）
                # SAM3 模型通常有 image_encoder 或類似屬性
                if hasattr(model, 'image_encoder'):
                    encoder = model.image_encoder
                elif hasattr(model, 'encoder'):
                    encoder = model.encoder
                else:
                    # 如果沒有明確的 encoder，假設整個模型就是 encoder
                    encoder = model
                
                return encoder
            
            # 回退到通用載入（例如 SAM2 或其他模型）
            else:
                print(f"Loading generic model: {model_name}")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name)
                return model
            
        except Exception as e:
            print(f"Error loading SAM3 encoder: {e}")
            print("Trying fallback: simple Vision Transformer")
            
            # 最終回退：使用簡單的 Vision Transformer
            try:
                import timm
                encoder = timm.create_model('vit_large_patch14_clip_336', pretrained=True)
                return encoder
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise
    
    def _build_sam3_model(self):
        """Build a basic SAM3 model structure for loading checkpoints.
        
        注意：這是一個簡化的佔位符，實際上應該使用官方的 build_sam3_image_model。
        如果有本地 checkpoint，建議直接使用官方 API。
        """
        try:
            from sam3.model_builder import build_sam3_image_model
            return build_sam3_image_model()
        except ImportError:
            print("SAM3 package not available. Using placeholder architecture...")
            # 回退架構
            import timm
            return timm.create_model('vit_large_patch14_clip_336', pretrained=False)
    
    def forward(self, x):
        """Forward pass through the encoder."""
        return self.encoder(x)