import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with LN → MHSA → MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        # Multi-head self-attention
        x_res = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x_res + attn_out

        # MLP
        x_res = x
        x = self.norm2(x)
        x = x_res + self.mlp(x)
        return x


class StackedHierarchicalTransformer(nn.Module):
    """
    Implements the EXACT hierarchy you required:

    Block1: concat([feature_map_embed, st_encode])
    Block2: concat([Block1_out, st_encode])
    Block3: concat([Block2_out, Block1_out])
    Block4: concat([Block3_out, Block2_out])
    Block5: concat([Block4_out, Block3_out])
    """

    def __init__(self, embed_dim=256, num_heads=8, num_blocks=5):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        # Each block must expand to accept concatenated embeddings
        # So we allow dynamic input dim
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])

        # projections to match concatenation dims
        self.proj_in = nn.Linear(embed_dim * 2, embed_dim)   # for Block1 & Block2
        self.proj_mid = nn.Linear(embed_dim * 2, embed_dim)  # for Block3–5

    def forward(self, feature_map_embed, st_embed):
        """
        feature_map_embed : (B, D)
        st_embed          : (B, D)

        Both become seq length 1: (B, 1, D)
        """

        B, D = st_embed.shape
        f1 = feature_map_embed.unsqueeze(1)   # (B,1,D)
        s1 = st_embed.unsqueeze(1)            # (B,1,D)

        # ----------------------------------------------------
        # BLOCK 1
        # input_1 = concat([feature_map_embed , spatiotemporal_encoding])
        # ----------------------------------------------------
        inp1 = torch.cat([f1, s1], dim=-1)               # (B,1,2D)
        inp1 = self.proj_in(inp1)                        # → (B,1,D)
        out1 = self.blocks[0](inp1)                      # (B,1,D)

        # ----------------------------------------------------
        # BLOCK 2
        # input_2 = concat([output_1 , spatiotemporal_encoding])
        # ----------------------------------------------------
        inp2 = torch.cat([out1, s1], dim=-1)             # (B,1,2D)
        inp2 = self.proj_in(inp2)
        out2 = self.blocks[1](inp2)                      # (B,1,D)

        # ----------------------------------------------------
        # BLOCK 3
        # input_3 = concat([output_2 , output_1])
        # ----------------------------------------------------
        inp3 = torch.cat([out2, out1], dim=-1)
        inp3 = self.proj_mid(inp3)
        out3 = self.blocks[2](inp3)

        # ----------------------------------------------------
        # BLOCK 4
        # input_4 = concat([output_3 , output_2])
        # ----------------------------------------------------
        inp4 = torch.cat([out3, out2], dim=-1)
        inp4 = self.proj_mid(inp4)
        out4 = self.blocks[3](inp4)

        # ----------------------------------------------------
        # BLOCK 5
        # input_5 = concat([output_4 , output_3])
        # ----------------------------------------------------
        inp5 = torch.cat([out4, out3], dim=-1)
        inp5 = self.proj_mid(inp5)
        out5 = self.blocks[4](inp5)

        # Final output → SAC/MLP head
        out_final = out5.squeeze(1)  # (B, D)

        return out_final, (out1, out2, out3, out4, out5)


class TimeSformerBlock(nn.Module):
    """
    A single block of Divided Space-Time Attention.
    1. Temporal Attention: (B*S, T, D) - Mixing time info per pixel
    2. Spatial Attention: (B*T, S, D) - Mixing spatial info per frame
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Temporal Attention Components
        self.norm_time = nn.LayerNorm(dim)
        self.attn_time = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Spatial Attention Components
        self.norm_space = nn.LayerNorm(dim)
        self.attn_space = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed Forward Network
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, spatial_size):
        """
        x: Input tensor (Batch, Time, Spatial_Pixels, Dim)
        """
        B, T, S, D = x.shape
        
        # --- 1. TEMPORAL ATTENTION ---
        # Reshape to merge Batch and Space: We treat (Batch * Pixel) as independent sequences of Time
        xt = x.permute(0, 2, 1, 3).reshape(B * S, T, D) 
        
        # Apply Attention over Time
        # xt is (Batch*Space, Time, Dim)
        res_t = xt
        xt = self.norm_time(xt)
        xt, _ = self.attn_time(xt, xt, xt)
        xt = xt + res_t
        
        # Reshape back to (B, T, S, D)
        x = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)

        # --- 2. SPATIAL ATTENTION ---
        # Reshape to merge Batch and Time: We treat (Batch * Time) as independent images
        xs = x.reshape(B * T, S, D)
        
        # Apply Attention over Space
        res_s = xs
        xs = self.norm_space(xs)
        xs, _ = self.attn_space(xs, xs, xs)
        xs = xs + res_s
        
        # Reshape back to (B, T, S, D)
        x = xs.reshape(B, T, S, D)

        # --- 3. FEED FORWARD ---
        x = x + self.ffn(self.norm_ffn(x))
        
        return x

class SpatioTemporalEncoder(nn.Module):
    def __init__(
        self, 
        img_size=7,               # Input Spatial dimension (e.g. 7x7 from ResNet)
        in_channels=2048,         # Input channels from CNN
        embed_dim=512,            # Internal Transformer Dimension
        num_frames=8,             # Length of history sequence
        num_heads=8,
        num_layers=4,
        num_classes=0,            # 0 for latent representation output
        dropout=0.1
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.H = img_size[1]
        self.W = img_size[0]
        self.num_patches = self.H * self.W

        # 1. Input Projection (CNN Channels -> Transformer Dim)
        # We use a 1x1 conv to project 2048 dim to 512 dim
        self.proj = nn.Linear(in_channels, embed_dim)

        # 2. Positional Embeddings
        # CLS Token: A learnable vector to aggregate the final representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Spatial Position Embedding: (1, 1, S, D) - Broadcasts across Time/Batch
        self.pos_embed_spatial = nn.Parameter(torch.randn(1, 1, self.num_patches, embed_dim))
        
        # Temporal Position Embedding: (1, T, 1, D) - Broadcasts across Space/Batch
        self.pos_embed_temporal = nn.Parameter(torch.randn(1, num_frames, 1, embed_dim))
        
        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 4. Final Norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Input x: (Batch, Time, Channels, Height, Width)
                 e.g., (B, 8, 2048, 7, 7)
        Returns: (Batch, embed_dim) -> The rich latent context vector
        """
        B, T, C, H, W = x.shape
        
        # --- Preprocessing ---
        # Flatten spatial dims: (B, T, C, H, W) -> (B, T, H*W, C)
        x = x.flatten(3).permute(0, 1, 3, 2) 
        
        # Project features: (B, T, S, C) -> (B, T, S, D)
        x = self.proj(x)
        
        # --- Add Positional Encodings ---
        # Add Spatial PE (Broadcasts over Time)
        x = x + self.pos_embed_spatial
        
        # Add Temporal PE (Broadcasts over Space)
        x = x + self.pos_embed_temporal

        # --- Transformer Layers ---
        for block in self.blocks:
            x = block(x, spatial_size=(H, W))

        # --- Output Aggregation ---
        # We need to collapse this rich 4D tensor (B, T, S, D) into a 1D vector (B, D).
        # Standard approach: Global Average Pooling over Space and Time
        
        # 1. Mean over Space (S) -> (B, T, D)
        x_spatial_mean = torch.mean(x, dim=2)
        
        # 2. Mean over Time (T) -> (B, D)
        x_final = torch.mean(x_spatial_mean, dim=1)
        
        # Final Norm
        x_final = self.norm(x_final)
        
        return x_final