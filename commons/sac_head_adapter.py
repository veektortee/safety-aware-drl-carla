# commons/sac_head_adapter.py
import torch.nn as nn
import torch as th

class SACHeadAdapter(nn.Module):
    """
    Optional helper MLP to convert the transformer's output to the
    dimensions expected by the Actor and Critic nets.
    Not strictly necessary if Actor/Critic net_arch already match embed_dim,
    but useful if you want to map embed_dim -> features_dim (e.g. 512 -> 256).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if in_dim == out_dim:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)
