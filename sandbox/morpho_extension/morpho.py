import torch
import torch.nn as nn
import morpho_cuda

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
