import torch
import torch.nn as nn
import mlp_cuda


class MLPFunction(torch.autograd.Function):
    """
    Custom autograd Function that calls our C++/CUDA routines:
     - forward_all(...) => returns (out1, out2)
     - backward_all(...) => returns (grad_x, grad_w1, grad_b1, grad_w2, grad_b2)
    """
    @staticmethod
    def forward(ctx, input_, w1, b1, w2, b2):
        """
        We do a 2-layer MLP with ReLU:
          out1 = ReLU(input @ w1 + b1)
          out2 = ReLU(out1 @ w2 + b2)
        We return out2, but we also keep out1 around for the backward pass.
        """
        # Call into our custom forward that returns both layer outputs
        out1, out2 = mlp_cuda.forward_all(input_, w1, b1, w2, b2)
        # Save everything we need for backward
        ctx.save_for_backward(input_, out1, out2, w1, b1, w2, b2)
        return out2

    @staticmethod
    def backward(ctx, grad_out2):
        """
        We compute the gradients wrt:
          - input
          - w1, b1
          - w2, b2
        and return them in order.
        """
        input_, out1, out2, w1, b1, w2, b2 = ctx.saved_tensors

        # Call our custom backward
        grads = mlp_cuda.backward_all(grad_out2, input_, out1, out2, w1, b1, w2, b2)
        # grads is a tuple of 5 tensors: (grad_x, grad_w1, grad_b1, grad_w2, grad_b2)
        grad_x, grad_w1, grad_b1, grad_w2, grad_b2 = grads

        return grad_x, grad_w1, grad_b1, grad_w2, grad_b2


class MLP(nn.Module):
    """
    2-layer MLP that uses the above custom forward/backward for training.
    """
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        # Register the parameters just like normal
        self.weights1 = nn.Parameter(
            0.01 * torch.randn(in_features, hidden_dim, device='cuda')
        )
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim, device='cuda'))

        self.weights2 = nn.Parameter(
            0.01 * torch.randn(hidden_dim, out_features, device='cuda')
        )
        self.bias2 = nn.Parameter(torch.zeros(out_features, device='cuda'))

    def forward(self, x):
        # Use our custom MLPFunction
        return MLPFunction.apply(x, self.weights1, self.bias1, self.weights2, self.bias2)
