import torch
import torch.nn as nn
from typing import Tuple

import greyscale_morphology_extension


class Dilation2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, origin: Tuple[int, int], padding_value: float):
        """
        input: [N, Cin, H, W]
        weight: [Cout, Cin, Kh, Kw]
        Returns: output: [N, Cout, H, W]
        """
        # Call C++/CUDA forward
        out, argmax = greyscale_morphology_extension.dilation2d_forward(
            input, weight, origin[0], origin[1], padding_value
        )

        # Save for backward
        ctx.save_for_backward(input, weight, argmax)
        ctx.origin = origin
        ctx.padding_value = padding_value
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        grad_out: [N, Cout, Hout, Wout]
        Returns: (grad_input, grad_weight)
        """
        input, weight, argmax = ctx.saved_tensors
        origin = ctx.origin
        padding_value = ctx.padding_value

        # Call C++/CUDA backward
        grad_in, grad_w = greyscale_morphology_extension.dilation2d_backward(
            grad_out, argmax, input, weight, origin[0], origin[1], padding_value
        )

        return grad_in, grad_w, None, None


class Dilation2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, origin=(1, 1), padding_value=-float('inf')):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.origin = origin
        self.padding_value = padding_value
        
        # Initialize weights randomly
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *kernel_size))
        
    def forward(self, x):
        return Dilation2DFunction.apply(x, self.weight, self.origin, self.padding_value)

