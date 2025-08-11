import torch
import torch.nn as nn
from typing import List, Dict


class RoPENd(torch.nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    def __init__(self, shape, base=10000):
        super(RoPENd, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0, f'shape[-1] ({feature_dim}) is not divisible by 2 * len(shape[:-1]) ({2 * len(channel_dims)})'

        # tensor of angles to use
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

        # create a stack of angles multiplied by position
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)

        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)

        # store in a buffer so it can be saved in model parameters
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        # convert input into complex numbers to perform rotation
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = self.rotations * x
        return torch.view_as_real(pe_x).flatten(-2)
    

class miniViT_3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(miniViT_3D, self).__init__()
        # rope = RoPENd((in_channels, 64, 64, 64))  # Example shape

    def forward(self, x: torch.Tensor, bit_depth_tensor: torch.Tensor) -> torch.Tensor:
        # NOTE: bit_depth_tensor specifies the bit depth of each frame in the input tensor x - 
        # This value should be concat with the rotary positional embedding

        # x is a 4D tensor with shape (batch_size, video_frame, channels, height, width)
        
        return x  # Placeholder