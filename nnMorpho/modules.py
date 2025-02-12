import torch
from torch import nn, Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import Optional, Union, Tuple

# import morphological_dilation2d

_type_to_pair = Union[int, Tuple[int, int]]


class Dilation2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, origin: Tuple[int, int], padding_value: float):
        """
        input: [N, Cin, H, W]
        weight: [Cout, Cin, Kh, Kw]
        Returns: output: [N, Cout, H, W]
        """
        # Call C++/CUDA forward
        out, argmax = morphological_dilation2d.morphological_dilation2d_forward(
            input, weight, origin[0], origin[1], padding_value
        )

        # Save for backward
        ctx.save_for_backward(input, weight, argmax)
        ctx.origin = origin
        ctx.padding_value = padding_value
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        """
        grad_out: [N, Cout, Hout, Wout]
        Returns: (grad_input, grad_weight)
        """
        input, weight, argmax = ctx.saved_tensors
        origin = ctx.origin
        padding_value = ctx.padding_value

        # Call C++/CUDA backward
        grad_in, grad_w = morphological_dilation2d.morphological_dilation2d_backward(
            grad_out, argmax, input, weight, origin[0], origin[1], padding_value
        )

        return grad_in, grad_w, None


class Dilation2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: _type_to_pair,
            origin: Optional[_type_to_pair] = None,
            padding_value: float = -float('inf'),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)

        # Default origin is kernel_size // 2 if not provided
        if origin is None:
            self.origin = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        else:
            self.origin = origin

        self.padding_value = padding_value

        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, *self.kernel_size)
        )

    def forward(self, input: Tensor) -> Tensor:
        return Dilation2dFunction.apply(
            input,
            self.weight,
            self.origin,
            self.padding_value
        )
