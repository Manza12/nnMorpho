import torch
import torch.nn as nn
import torch.optim as optim

from mlp import MLP

# Just a standard 2-layer MLP in PyTorch
class ClassicalMLP(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

def main():
    device = "cuda"

    # Random problem: input=16 dims, hidden=32, output=8 dims
    B = 10
    in_features = 16
    hidden_dim = 32
    out_features = 8

    # Create random input & target
    x = torch.randn(B, in_features, device=device)
    target = torch.randn(B, out_features, device=device)

    # Classical
    classical_mlp = ClassicalMLP(in_features, hidden_dim, out_features).to(device)
    classical_optim = optim.SGD(classical_mlp.parameters(), lr=1e-2)

    # Custom
    custom_mlp = MLP(in_features, hidden_dim, out_features).to(device)
    # Note: Because our custom kernel does not return param grads,
    # PyTorch won't automatically compute them from this raw kernel.
    # We'll still define an optimizer for demonstration:
    params = [custom_mlp.weights1, custom_mlp.bias1,
              custom_mlp.weights2, custom_mlp.bias2]
    custom_optim = optim.SGD(params, lr=1e-2)

    # Forward/backward for classical
    out_classical = classical_mlp(x)
    loss_classical = ((out_classical - target)**2).mean()
    classical_optim.zero_grad()
    loss_classical.backward()  # compute param grads
    classical_optim.step()

    # Forward/backward for custom
    out_custom = custom_mlp(x)
    loss_custom = ((out_custom - target)**2).mean()
    custom_optim.zero_grad()
    loss_custom.backward()  # but param grads remain None with raw kernel
    custom_optim.step()

    print("Classical MLP loss =", loss_classical.item())
    print("Custom MLP loss    =", loss_custom.item())

if __name__ == "__main__":
    main()
