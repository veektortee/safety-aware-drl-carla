"""
Pipeline module that composes the FeatureExtractor, SpatioTemporalEncoder
and a stacked transformer (either StackedHierarchicalTransformer or
StackedTimeSformer) into a single pipeline object ready for use in
training the SAC agent.

Usage:
    from models.pipeline import Pipeline

    pipe = Pipeline.from_defaults(fe_weights_path="./feature_extractor.pth", use_timesformer=True)
    final_emb, cnn_emb = pipe.process_sequence(list_of_frames)

    # Or build an SB3-compatible feature extractor for direct use in policies
    tf_extractor = pipe.build_transformer_feature_extractor(observation_space)

"""
from typing import Optional, Sequence
import torch
import torch.nn as nn
import numpy as np

from commons.feature_extractor import FeatureExtractor
from commons.spatioTemporal_transformer import SpatioTemporalEncoder
from commons.spatioTemporal_transformer import StackedHierarchicalTransformer, StackedTimeSformer
from commons.transformer_feature_extractor import TransformerFeatureExtractor


class Pipeline:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        st_encoder: SpatioTemporalEncoder,
        stacked_transformer: nn.Module,
        device: str = "cpu",
        in_channels: int = 2048,
        embed_dim: int = 512,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device)
        self.st_encoder = st_encoder.to(self.device)
        self.stacked_transformer = stacked_transformer.to(self.device)

        # small projection to map CNN feature maps (C,H,W) -> embed_dim
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.cnn_proj = nn.Linear(in_channels, embed_dim).to(self.device)

    @staticmethod
    def from_defaults(
        num_frames: int = 8,
        embed_dim: int = 512,
        use_timesformer: bool = False,
        fe_weights_path: Optional[str] = None,
        device: str = "cpu",
    ) -> "Pipeline":
        # Build components with sensible defaults
        fe = FeatureExtractor(device=device)
        if fe_weights_path is not None:
            state = torch.load(fe_weights_path, map_location=device)
            # allow both checkpoint dict and raw state_dict
            if isinstance(state, dict) and "model_state" in state:
                state_dict = state["model_state"]
            elif isinstance(state, dict) and any(k.startswith("trunk") for k in state.keys()):
                state_dict = state
            else:
                state_dict = state
            try:
                fe.trunk.load_state_dict(state_dict)
            except Exception:
                # best-effort: try loading into entire module
                try:
                    fe.load_state_dict(state_dict)
                except Exception:
                    print("Warning: could not load feature extractor weights from", fe_weights_path)

        st_enc = SpatioTemporalEncoder(
            img_size=(7, 7),
            in_channels=2048,
            embed_dim=embed_dim,
            num_frames=num_frames,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
        )

        if use_timesformer:
            stacked = StackedTimeSformer(embed_dim=embed_dim, num_heads=8, num_blocks=5)
        else:
            stacked = StackedHierarchicalTransformer(embed_dim=embed_dim, num_heads=8, num_blocks=5)

        return Pipeline(feature_extractor=fe, st_encoder=st_enc, stacked_transformer=stacked, device=device,
                        in_channels=2048, embed_dim=embed_dim)

    def process_sequence(self, frames: Sequence[np.ndarray]):
        """
        Process a temporal sequence of raw cv2 frames (list/array of T frames HxWx3 BGR)
        and return the final stacked-transformer embedding + cnn embedding.

        Args:
            frames: Sequence of T numpy frames (H, W, 3) in BGR format
        Returns:
            out_final: torch.Tensor shape (D,) final embedding
            cnn_embed: torch.Tensor shape (D,) embedding derived from latest frame's CNN feature map
        """
        # Validate input
        if len(frames) == 0:
            raise ValueError("frames sequence is empty")

        # Extract per-frame feature maps
        feat_maps = []
        for f in frames:
            fmap = self.feature_extractor.extract_feature_map(f)  # (C,H,W) on device
            feat_maps.append(fmap)

        # Stack into tensor (T, C, H, W) and add batch dim -> (1, T, C, H, W)
        seq = torch.stack(feat_maps, dim=0).unsqueeze(0).to(self.device)

        # Get spatio-temporal encoding (B, D)
        st_out = self.st_encoder(seq)  # returns (B, D)

        # CNN embed from the last frame's feature map
        last_fmap = feat_maps[-1].unsqueeze(0)  # (1, C, H, W)
        pooled = self.cnn_pool(last_fmap)  # (1, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (1, C)
        cnn_embed = self.cnn_proj(pooled.to(self.cnn_proj.weight.device))  # (1, D)

        # Pass through stacked transformer
        out_final, intermediates = self.stacked_transformer(cnn_embed, st_out)

        # Squeeze batch dim for return
        return out_final.squeeze(0), cnn_embed.squeeze(0)

    def build_transformer_feature_extractor(self, observation_space):
        """
        Build an SB3-compatible TransformerFeatureExtractor that uses the
        pipeline's ResNet trunk and SpatioTemporalEncoder. Useful to pass
        directly as `policy_kwargs={'features_extractor_class': tf_extractor}`
        when instantiating SB3 policies.
        """
        # Reuse the existing encoder and trunk
        # TransformerFeatureExtractor expects `st_encoder` instance and optional backbone
        tf_extractor = TransformerFeatureExtractor(
            observation_space=observation_space,
            st_encoder=self.st_encoder,
            backbone=self.feature_extractor.trunk,
            device=str(self.device),
            embed_dim=self.st_encoder.embed_dim,
        )
        return tf_extractor
