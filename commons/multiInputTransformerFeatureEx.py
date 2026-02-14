import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from commons.transformer_feature_extractor import TransformerFeatureExtractor

class TransformerMultiInputExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible feature extractor for SAC with:
    - SpatioTemporal vision encoder
    - Ego / state MLP
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        transformer_extractor,          # your TransformerFeatureExtractor
        ego_dim: int,
        ego_out_dim: int = 128,
    ):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.transformer = transformer_extractor.to(self.device)
        self.transformer.eval()  # start frozen (recommended)

        self.ego_net = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.ReLU(),
            nn.Linear(64, ego_out_dim),
            nn.ReLU(),
        ).to(self.device)

        features_dim = (
            self.transformer._adapter_dim + ego_out_dim
        )

        super().__init__(observation_space, features_dim)

    def forward(self, observations):
        """
        observations:
        {
            "rgb":  (B, T, 3, H, W),
            "ego":  (B, ego_dim)
        }
        """
        rgb = observations["rgb"].float().to(self.device) / 255.0
        ego = observations["ego"].float().to(self.device)

        with th.no_grad():
            visual_feat = self.transformer(rgb)   # (B, Dv)

        ego_feat = self.ego_net(ego)              # (B, De)

        return th.cat([visual_feat, ego_feat], dim=1)
