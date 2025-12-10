import torch
import torch.nn as nn
import math

class PhysicalTimeEncoder(nn.Module):
    def __init__(self, d_model=256, max_period=10000.0, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.device = device
        
        half_dim = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, 2, device=device).float() * -(math.log(max_period) / half_dim)
        )
        self.register_buffer('div_term', div_term)
        self.to(device)

    def forward(self, time_val):
        """
        time_val: Scalar or tensor of actual time (seconds)
        Returns: (1, d_model)
        """
        # Make sure input is a tensor on the correct device
        if not torch.is_tensor(time_val):
            time_val = torch.tensor([time_val], device=self.device)
            
        # Reshape to (Batch, 1) for broadcasting
        t = time_val.view(1, 1) 

        scaled_time = t * self.div_term 
        pe_sin = torch.sin(scaled_time)
        pe_cos = torch.cos(scaled_time)
        
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe # (1, d_model)