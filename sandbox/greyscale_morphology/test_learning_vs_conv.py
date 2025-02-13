import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from greyscale_morphology import Dilation2DFunction, Dilation2D

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Ground Truth Setup ---
# The GT kernel used for generating targets.
w_gt = torch.tensor([[[[0, 2, 1],
                       [1, 0, 3],
                       [1, 2, 0]]]], dtype=torch.float).to(device)
origin = (1, 1)
padding_value = -float('inf')

# --- Hyperparameters ---
num_epochs = 500
num_samples = 10   # mini-batches per epoch
batch_size = 10    # samples per mini-batch
loss_fn = nn.MSELoss()

# --- Dilation2D Model ---
model_dilation = Dilation2D(
    in_channels=1,
    out_channels=1,
    kernel_size=w_gt.shape[-2:],
    origin=origin,
    padding_value=padding_value
).to(device)
optimizer_dilation = optim.Adam(model_dilation.parameters(), lr=0.01)

# --- Conv2d Model ---
# We use padding='same' so that output shape matches input shape.
model_conv = nn.Conv2d(in_channels=1, out_channels=1,
                       kernel_size=w_gt.shape[-2:], bias=False, padding='same').to(device)
optimizer_conv = optim.Adam(model_conv.parameters(), lr=0.01)

# --- Training Loop for Dilation2D Model ---
print("Training Dilation2D model...")
start_time = time.time()
for epoch in range(num_epochs):
    optimizer_dilation.zero_grad()
    epoch_loss = 0.0
    for _ in range(num_samples):
        # Generate random input (using a 4x4 spatial size)
        x = torch.randn(batch_size, 1, 4, 4).to(device)
        # Target: output from the GT dilation operator
        with torch.no_grad():
            y_target = Dilation2DFunction.apply(x, w_gt, origin, padding_value)
        y_pred = model_dilation(x)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        epoch_loss += loss.item()
    optimizer_dilation.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} (Dilation2D): loss = {epoch_loss/num_samples:.6f}")
dilation_time = time.time() - start_time
print("Dilation2D training time: {:.4f} seconds".format(dilation_time))


# --- Training Loop for Conv2d Model ---
print("\nTraining Conv2d model...")
start_time = time.time()
for epoch in range(num_epochs):
    optimizer_conv.zero_grad()
    epoch_loss = 0.0
    for _ in range(num_samples):
        x = torch.randn(batch_size, 1, 28, 28).to(device)
        # Target: output from the GT conv operator (using F.conv2d with 'same' padding)
        with torch.no_grad():
            y_target = F.conv2d(x, w_gt, bias=None, padding='same')
        y_pred = model_conv(x)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        epoch_loss += loss.item()
    optimizer_conv.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch} (Conv2d): loss = {epoch_loss/num_samples:.6f}")
conv_time = time.time() - start_time
print("Conv2d training time: {:.4f} seconds".format(conv_time))


# --- Compare Learned Kernels ---
print("\nGround Truth Kernel (w_gt):")
print(w_gt.cpu())

print("\nLearned Dilation2D Kernel:")
print(model_dilation.weight.data.cpu())

print("\nLearned Conv2d Kernel:")
print(model_conv.weight.data.cpu())

# --- Compare Learned kernels
print("\nComparing learned kernels...")
print("Dilation2D kernel difference: ", torch.norm(w_gt - model_dilation.weight.data))
print("Conv2d kernel difference: ", torch.norm(w_gt - model_conv.weight.data))
