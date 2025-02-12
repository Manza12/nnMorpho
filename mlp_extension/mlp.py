import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Load the extension
mlp_cuda = load(name="mlp_cuda", sources=["mlp_cuda.cpp", "mlp_cuda_kernel.cu"], verbose=True)

class MLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights1, bias1, weights2, bias2):
        output = mlp_cuda.forward(input, weights1, bias1, weights2, bias2)
        ctx.save_for_backward(input, weights1, weights2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights1, weights2 = ctx.saved_tensors
        grad_input = mlp_cuda.backward(grad_output, input, weights1, weights2)
        return grad_input, None, None, None, None  # Gradients for inputs only

class MLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(MLP, self).__init__()
        self.weights1 = nn.Parameter(torch.randn(in_features, hidden_dim, device="cuda"))
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim, device="cuda"))
        self.weights2 = nn.Parameter(torch.randn(hidden_dim, out_features, device="cuda"))
        self.bias2 = nn.Parameter(torch.zeros(out_features, device="cuda"))

    def forward(self, x):
        return MLPFunction.apply(x, self.weights1, self.bias1, self.weights2, self.bias2)
