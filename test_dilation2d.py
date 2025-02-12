import torch
from nnMorpho.modules import Dilation2d

# Example: create a dilation layer with explicit parameters
layer = Dilation2d(in_channels=3, out_channels=6, kernel_size=3, origin=1, padding_value=-float('inf')).cuda()

x = torch.randn(2, 3, 8, 8, device='cuda')  # e.g. batch=2, channel_in=3
out = layer(x)
print("Output shape:", out.shape)
