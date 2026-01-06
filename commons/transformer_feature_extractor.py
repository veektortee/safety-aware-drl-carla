# commons/transformer_feature_extractor.py
# SB3-compatible features extractor that wraps:
#   - ResNet trunk (children()[:-2]) to produce (B*T, 2048, 7, 7)
#   - SpatioTemporalEncoder to produce final (B, embed_dim)
#
# Requirements:
#   - observation passed to policy must be a tensor of shape (B, T, C, H, W)
#     (SB3 will pass numpy arrays which are converted to torch tensors by the VecEnv wrappers).
#   - Provide an instance of SpatioTemporalEncoder (st_encoder) when creating this extractor.

from typing import Dict, Any, Optional

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sac_head_adapter import SACHeadAdapter
from torchvision import models
import numpy as np

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    BaseFeaturesExtractor wrapper for SpatioTemporalEncoder.

    Args:
        observation_space: gym space (ignored here, but kept for API compatibility)
        st_encoder: an instance of your SpatioTemporalEncoder (expects input shape (B, T, C, H, W))
        backbone: optional ResNet trunk to use (if None, we build resnet50 children()[:-2])
        embed_dim: final output dim (must match st_encoder output_dim)
        device: 'cpu' or 'cuda'
    """

    def __init__(
        self,
        observation_space,
        st_encoder: nn.Module,
        backbone: Optional[nn.Module] = None,
        device: str = "cpu",
        normalize_images: bool = True,
        embed_dim: int = 512,
        adapter_dim: int = 256
    ):
        # features_dim must be set to the dimensionality returned by forward()
        super().__init__(observation_space, features_dim=embed_dim)
        self.device = th.device(device if th.cuda.is_available() else "cpu")
        
        # SpatioTemporalEncoder instance (should already be configured: img_size, in_channels, num_frames, embed_dim)
        self.st_encoder = st_encoder.to(self.device)
        self.st_encoder.eval()  # the encoder is used as a feature module in the extractor

        # ResNet trunk (children()[:-2]) producing (2048, 7, 7) for each frame
        if backbone is None:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            trunk_modules = list(resnet.children())[:-2]
            self.trunk = nn.Sequential(*trunk_modules).to(self.device)
        else:
            self.trunk = backbone.to(self.device)

        self.trunk.eval()
        
        self.adapter = SACHeadAdapter(
            in_dim=embed_dim,
            out_dim=adapter_dim
        ).to(self.device)
        
        self._embed_dim = embed_dim
        self._adapter_dim = adapter_dim
        
        # infer expected channels & resolution from trunk: we assume ResNet behavior
        self.in_channels = 3

        # keep embed_dim in the class
        self._embed_dim = embed_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        observations (torch.Tensor): expected shape (B, T, C, H, W)
        returns: tensor (B, embed_dim)
        """
        # Validate shape
        if not (observations is not None and observations.ndim == 5):
            # try to support (B, C, H, W) single frame repeated T times -> expand
            if observations.ndim == 4:
                # assume single frame per batch, expand time dimension to 1
                observations = observations.unsqueeze(1)  # (B, 1, C, H, W)
            else:
                raise ValueError("TransformerFeatureExtractor expects input shape (B, T, C, H, W) or (B, C, H, W)")

        B, T, C, H, W = observations.shape
        # Move to extractor device
        obs = observations.to(self.device)

        # Efficiently run backbone on flattened batch: (B*T, C, H, W)
        flat = obs.view(B * T, C, H, W)
        with th.no_grad():
            trunk_feats = self.trunk(flat)  # shape: (B*T, 2048, H2, W2) e.g. (B*T, 2048, 7, 7)

        # Make sure trunk_feats are float and on device
        trunk_feats = trunk_feats.to(self.device)

        # Reshape to (B, T, C_trunk, H2, W2)
        _, Ctr, H2, W2 = trunk_feats.shape
        trunk_feats = trunk_feats.view(B, T, Ctr, H2, W2)

        # Now pass the sequence through the SpatioTemporalEncoder (expects (B, T, C, H, W))
        # st_encoder should already be on the right device
        with th.no_grad():
            st_out = self.st_encoder(trunk_feats)  # expected (B, embed_dim) or (B, T, embed_dim) depending on st_encoder design

        # If the encoder returns a per-frame sequence (B, T, D), take last timestep (most recent)
        if st_out.ndim == 3:
            # prefer the last time step as the feature (B, D)
            st_out = st_out[:, -1, :]

        # ensure dtype and device
        st_out = st_out.to(self.device).float()
        
        st_out = self.adapter(st_out)


        # st_out is (B, embed_dim) as required by SB3
        return st_out

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        # This allows SB3 to re-create it when loading.
        return {
            "st_encoder": self.st_encoder,
            "embed_dim": self._embed_dim,
            "device": str(self.device)
        }
