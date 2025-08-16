import torch
import torch.nn as nn

# SpatioTemporal Attention
class LocalCubeletAttention(nn.Module):  # b c t h w
    def __init__(self):
        super(LocalCubeletAttention, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=4, 
                               kernel_size=7, padding=7//2, bias=False)
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
        self.conv1 = nn.Linear(in_features=2, out_features=1, bias=False)
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
        self.local_attention = LocalCubeletAttention()  # 3D conv attention (B, C, T, H, W)
        self.temporal_attention = TemporalLocalAttention()  # temporal only (B, T, C)
    
    def forward(self, x):  # x shape: [B, T, C, H, W]
        # Permute to (B, C, T, H, W) for 3D conv
        x = x.permute(0, 2, 1, 3, 4)
        x = self.local_attention(x)
        x = x.permute(0, 2, 1, 3, 4)  # back to [B, T, C, H, W]

        # Spatial pooling for temporal attention
        x_pooled = x.mean(dim=[3,4])  # [B, T, C]

        # Temporal attention to smooth across frames
        x_temporal = self.temporal_attention(x_pooled)  # [B, T, C]

        # Broadcast temporal weights back to spatial dims
        x_temporal = x_temporal.unsqueeze(-1).unsqueeze(-1)  # [B, T, C, 1, 1]

        # Multiply original x by temporal attention weights
        x = x * x_temporal

        return x
