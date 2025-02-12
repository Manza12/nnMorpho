import torch
from torch import nn, Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from typing import Optional, Union, Tuple

import morphological_dilation2d


class Dilation2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor],
                padH: int, padW: int, useNegInfPad: bool):
        """
        input: [N, Cin, H, W]
        weight: [Cout, Cin, Kh, Kw]
        bias: [Cout] or None
        Returns: output: [N, Cout, Hout, Wout]
        """
        # Call C++/CUDA forward
        out, argmax = morphological_dilation2d.morphological_dilation2d_forward(
            input, weight, padH, padW, useNegInfPad
        )
        # If bias is present, broadcast-add: bias[c] across spatial
        if bias is not None:
            out += bias.view(1, -1, 1, 1)

        # Save for backward
        ctx.save_for_backward(input, weight, bias, argmax)

        ctx.padH = padH
        ctx.padW = padW
        ctx.useNegInfPad = useNegInfPad
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        """
        grad_out: [N, Cout, Hout, Wout]
        Returns: (grad_input, grad_weight, grad_bias)
        """
        input, weight, bias, argmax = ctx.saved_tensors
        padH = ctx.padH
        padW = ctx.padW
        useNegInfPad = ctx.useNegInfPad

        # 1) morpho backward to get grad_in, grad_w
        grad_in, grad_w = morphological_dilation2d.morphological_dilation2d_backward(
            grad_out, argmax, input, weight, padH, padW, useNegInfPad
        )

        # 2) grad_bias
        grad_bias = None
        if bias is not None:
            grad_bias = grad_out.sum(dim=[0, 2, 3])

        return grad_in, grad_w, grad_bias, None, None, None


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
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="neg_inf",  # or "neg_inf"
            device=None,
            dtype=None,
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        if isinstance(padding, str):
            padding_ = padding
        else:
            padding_ = _pair(padding)
        dilation_ = _pair(dilation)

        super().__init__(
            in_channels, out_channels,
            kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode,
            device=device, dtype=dtype,
        )

        # Decide padH, padW from self.padding (which we stored in _ConvNd).
        # If we truly just want padH = padding_[0], padW = padding_[1], do:
        if isinstance(padding, tuple):
            self.padH, self.padW = padding
        else:
            # if it's int or str, handle accordingly
            if isinstance(padding, int):
                self.padH = padding
                self.padW = padding
            else:
                # e.g. "same" => you'd parse differently
                self.padH = 0
                self.padW = 0

        # Decide whether to use neg-inf or 0 for out-of-bounds
        self.useNegInfPad = (padding_mode == "neg_inf")

    def _dilation_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        """
        Actually calls our custom autograd Function,
        with padH, padW, useNegInfPad as well.
        """
        return Dilation2dFunction.apply(
            input,
            weight,
            bias,
            self.padH,
            self.padW,
            self.useNegInfPad
        )

    def forward(self, input: Tensor) -> Tensor:
        return self._dilation_forward(input, self.weight, self.bias)
