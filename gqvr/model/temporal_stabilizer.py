import torch
import torch.nn as nn

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# SpatioTemporal Attention
class LocalCubeletAttention(nn.Module):  # b c t h w
    def __init__(self):
        super(LocalCubeletAttention, self).__init__()
        self.conv1 = zero_module(nn.Conv3d(in_channels=2, out_channels=4, 
                               kernel_size=7, padding=7//2, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        weight = torch.cat([max_out, avg_out], dim=1)
        weight = self.conv1(weight)

        out = self.sigmoid(weight) * x
        return out
    

# Temporal LIEM
class TemporalLocalAttention(nn.Module):  # b t c
    def __init__(self):
        super(TemporalLocalAttention, self).__init__()
        self.conv1 = zero_module(nn.Linear(in_features=2, out_features=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out, _ = torch.max(x, dim=-1, keepdim=True)
        avg_out = torch.mean(x, dim=-1, keepdim=True)

        weight = torch.cat([max_out, avg_out], dim=-1)
        weight = self.conv1(weight)

        out = self.sigmoid(weight) * x
        return out
    




# NOTE: 
# The 3D conv attention addresses small, local spatial-temporal noise/artifacts.
# The temporal-only module enforces smoothness along time globally, across all channels.
class TemporalConsistencyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_dtype = torch.float32                       # Yes, fixed & hardcoded at max precision since these are <3k params! Relax!
        self.local_attention = LocalCubeletAttention()          # 3D conv attention (B, C, T, H, W)
        self.temporal_attention = TemporalLocalAttention()      # Temporal only (B, T, C)
        self.gamma = nn.Parameter(torch.full((1, 1, 1, 1, 1), 0.1))   # Learnable residual scale; start at 0!

    def forward(self, x):  # x shape: [B, T, C, H, W]
        # Permute to (B, C, T, H, W) for 3D conv
        x_res = x.permute(0, 2, 1, 3, 4).to(self.weight_dtype)
        x_res = self.local_attention(x_res)
        x_res = x_res.permute(0, 2, 1, 3, 4)  # back to [B, T, C, H, W]

        # Spatial pooling for temporal attention
        x_pooled = x_res.mean(dim=[3,4])  # [B, T, C]

        # Temporal attention to smooth across frames
        x_temporal = self.temporal_attention(x_pooled)  # [B, T, C]

        # Broadcast temporal weights back to spatial dims
        x_temporal = x_temporal.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]

        # Multiply original x by temporal attention weights
        x = x * (1 + self.gamma * x_temporal)

        return x


if __name__ == "__main__":
    temp_module = TemporalConsistencyLayer()
    dummy_in = torch.randn(1, 64, 4, 64, 64, dtype=torch.float32)
    dummy_out = temp_module(dummy_in)
    print(dummy_out.size())