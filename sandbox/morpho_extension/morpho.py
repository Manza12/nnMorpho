import torch
import torch.nn as nn
import morpho_cuda


class MorphologicalFunction(torch.autograd.Function):
    """
    Single-layer morphological transform without bias:
      out[b,j] = max_i( inp[b,i] + weight[i,j] )
    We also store argmax for the backward pass.
    """

    @staticmethod
    def forward(ctx, inp, weight):
        # Call the extension's forward, which returns (out, argmax).
        out, argmax = morpho_cuda.morphological_forward(inp, weight)
        # Save for backward
        ctx.save_for_backward(inp, weight, argmax)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve saved tensors
        inp, weight, argmax = ctx.saved_tensors
        # Call the extension's backward, which returns (grad_in, grad_w)
        grad_inp, grad_w = morpho_cuda.morphological_backward(
            grad_out, inp, weight, argmax
        )
        return grad_inp, grad_w


class Morphological(nn.Module):
    """
    A nn.Module that mimics nn.Linear, but does:
      out[b,j] = max_i( x[b,i] + weight[i,j] )
    with no bias term.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # Register the morphological weight as a Parameter
        self.weight = nn.Parameter(
            0.01 * torch.randn(in_features, out_features)
        )

    def forward(self, x):
        # Use the custom autograd function
        return MorphologicalFunction.apply(x, self.weight)


class MorphoMLP(nn.Module):
    """A 2-layer Morphological Perceptron using morpho_cuda.Morphological + ReLU."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # Just like classical MLP, but use morphological layers.
        self.fc1 = morpho_cuda.Morphological(in_dim, hidden_dim)
        self.fc2 = morpho_cuda.Morphological(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
