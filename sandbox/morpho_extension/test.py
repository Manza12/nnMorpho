import torch
import torch.nn as nn
import torch.optim as optim

from morpho import MorphoMLP


class ClassicalMLP(nn.Module):
    """A plain 2-layer MLP using nn.Linear + ReLU just for comparison."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


def main():
    device = "cuda"
    B = 16
    in_dim, hidden_dim, out_dim = 10, 20, 5
    lr = 1e-2

    # Dummy input/target for a regression
    x = torch.randn(B, in_dim, device=device)
    target = torch.randn(B, out_dim, device=device)

    # 1) Classical MLP
    classical = ClassicalMLP(in_dim, hidden_dim, out_dim).to(device)
    opt_classical = optim.SGD(classical.parameters(), lr=lr)

    # 2) Morphological MLP
    morpho = MorphoMLP(in_dim, hidden_dim, out_dim).to(device)
    opt_morpho = optim.SGD(morpho.parameters(), lr=lr)

    for step in range(10):
        # Classical forward/backward
        out_cl = classical(x)
        loss_cl = (out_cl - target).pow(2).mean()
        opt_classical.zero_grad()
        loss_cl.backward()
        opt_classical.step()

        # Morphological forward/backward
        out_mo = morpho(x)
        loss_mo = (out_mo - target).pow(2).mean()
        opt_morpho.zero_grad()
        loss_mo.backward()
        opt_morpho.step()

        print(f"[Step {step}] Classical Loss = {loss_cl.item():.4f}, Morphological Loss = {loss_mo.item():.4f}")


if __name__ == "__main__":
    main()
