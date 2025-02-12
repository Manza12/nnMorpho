import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# ----------------------------------------------------------------------------
# Load the extension (compiles mlp_cuda.cpp + mlp_cuda_kernel.cu)
# ----------------------------------------------------------------------------
mlp_cuda = load(
    name="mlp_cuda",
    sources=["mlp_cuda.cpp", "mlp_cuda_kernel.cu"],
    verbose=True
)

class MLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights1, bias1, weights2, bias2):
        """
        Forward through 2-layer MLP:
            out1 = ReLU(input @ weights1 + bias1)
            out2 = ReLU(out1 @ weights2 + bias2)
        Returns out2.
        """
        # Call our custom CUDA forward
        output = mlp_cuda.forward(input, weights1, bias1, weights2, bias2)
        # Save tensors needed for backward
        ctx.save_for_backward(input, weights1, weights2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass. We will compute dL/dInput using our custom CUDA kernel
        (and return None for parameter gradients).
        """
        input_saved, weights1, weights2 = ctx.saved_tensors
        # Call our custom CUDA backward for input gradients
        grad_input = mlp_cuda.backward(grad_output, input_saved, weights1, weights2)
        # Return shape-matching "None" for parameters. So we do *not* train them
        # via custom kernel. They won't get gradients from our kernel alone!
        return grad_input, None, None, None, None


class MLP(nn.Module):
    """
    Custom MLP that uses the custom forward/backward GPU kernels.
    """
    def __init__(self, in_features, hidden_dim, out_features):
        super(MLP, self).__init__()
        self.weights1 = nn.Parameter(torch.randn(in_features, hidden_dim, device="cuda") * 0.01)
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim, device="cuda"))
        self.weights2 = nn.Parameter(torch.randn(hidden_dim, out_features, device="cuda") * 0.01)
        self.bias2 = nn.Parameter(torch.zeros(out_features, device="cuda"))

    def forward(self, x):
        return MLPFunction.apply(x, self.weights1, self.bias1, self.weights2, self.bias2)
