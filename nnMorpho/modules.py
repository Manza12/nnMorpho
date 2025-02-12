import torch
from torch import nn, Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import Optional, Union, Tuple

import morphological_dilation2d


class Dilation2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        """
        input: [N, Cin, H, W]
        weight: [Cout, Cin, Kh, Kw]
        bias: [Cout] or None
        Returns: output: [N, Cout, Hout, Wout]
        """
        # Call C++/CUDA forward
        out, argmax = morphological_dilation2d.morphological_dilation2d_forward(input, weight)
        # If bias is present, broadcast-add: bias[c] across spatial
        if bias is not None:
            out += bias.view(1, -1, 1, 1)

        # Save for backward
        ctx.save_for_backward(input, weight, bias, argmax)
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        """
        grad_out: [N, Cout, Hout, Wout]
        Returns: (grad_input, grad_weight, grad_bias)
        """
        input, weight, bias, argmax = ctx.saved_tensors

        # 1) morpho backward to get grad_in, grad_w
        grad_in, grad_w = morphological_dilation2d.morphological_dilation2d_backward(
            grad_out, argmax, input, weight
        )

        grad_bias = None
        if bias is not None:
            # bias is shape [Cout], so each channel c gets the sum of grad_out[:,c,:,:].
            # This is simply a sum over N,Hout,Wout for each channel.
            # shape of grad_out is [N, Cout, Hout, Wout].
            grad_bias = grad_out.sum(dim=[0, 2, 3])  # sum over N,Hout,Wout

        return grad_in, grad_w, grad_bias


class Dilation2d(_ConvNd):
    r"""
    A 2D morphological dilation layer, analogous to nn.Conv2d but using dilation instead of convolution.

    Naive version:
     - Ignores stride/padding/dilation/groups from the parent _ConvNd for now.
     - Calls morphological_dilation2d (C++/CUDA) and adds bias if present.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        if isinstance(padding, str):
            padding_ = padding
        else:
            padding_ = _pair(padding)
        dilation_ = _pair(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,  # transposed=False
            _pair(0),  # output_padding=(0,0)
            groups,
            bias,
            padding_mode,
            device=device,
            dtype=dtype,
        )

        # self.weight is created by _ConvNd with shape [out_channels, in_channels//groups, *kernel_size_]
        # self.bias if bias=True => shape [out_channels]

    def _dilation_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        """
        Actually calls our custom autograd Function,
        which calls the morphological_dilation2d extension.
        """
        return Dilation2dFunction.apply(input, weight, bias)

    def forward(self, input: Tensor) -> Tensor:
        # For now, ignoring stride/padding/dilation logic in the naive approach,
        # we just do a direct morphological dilation.
        return self._dilation_forward(input, self.weight, self.bias)
