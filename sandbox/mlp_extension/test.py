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
    B = 16           # batch size
    in_features = 10
    hidden_dim = 32
    out_features = 6
    lr = 1e-2

    # Random input/target for a fake regression problem
    x = torch.randn(B, in_features, device=device)
    target = torch.randn(B, out_features, device=device)

    # Classical MLP
    classical_mlp = ClassicalMLP(in_features, hidden_dim, out_features).to(device)
    classical_opt = optim.SGD(classical_mlp.parameters(), lr=lr)

    # Custom MLP
    custom_mlp = MLP(in_features, hidden_dim, out_features).to(device)
    custom_opt = optim.SGD(custom_mlp.parameters(), lr=lr)

    # Train both for a few steps
    for step in range(10):
        # ---- Classical MLP ----
        out_cl = classical_mlp(x)
        loss_cl = (out_cl - target).pow(2).mean()

        classical_opt.zero_grad()
        loss_cl.backward()
        classical_opt.step()

        # ---- Custom MLP ----
        out_cu = custom_mlp(x)
        loss_cu = (out_cu - target).pow(2).mean()

        custom_opt.zero_grad()
        loss_cu.backward()
        custom_opt.step()

        print(f"[Step {step}] Classical Loss = {loss_cl.item():.4f}, Custom Loss = {loss_cu.item():.4f}")

if __name__ == "__main__":
    main()
