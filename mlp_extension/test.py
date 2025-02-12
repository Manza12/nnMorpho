import torch
from mlp import MLP

device = "cuda"
model = MLP(10, 20, 5).to(device)

x = torch.randn(16, 10, device=device)
output = model(x)
print("Output shape:", output.shape)
