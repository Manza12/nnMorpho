import torch
import torch.nn as nn
import torch.optim as optim
from greyscale_morphology import Dilation2DFunction, Dilation2D

# --- Setup Ground Truth and Model ---
# Ground truth kernel (for example, the one you obtained)
w_gt = torch.tensor([[[[0, 2, 1],
                       [1, 0, 3]]]], dtype=torch.float).cuda()
origin = (1, 1)
padding_value = -float('inf')

# The trainable dilation layer with randomly initialized kernel
model = Dilation2D(in_channels=1, out_channels=1, kernel_size=w_gt.shape[-2:], origin=origin, padding_value=padding_value).cuda()

# Use an optimizer (Adam) to update the kernel weights.
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# --- Training Loop ---
num_epochs = 500
num_samples = 10  # number of random inputs per epoch

for epoch in range(num_epochs):
    optimizer.zero_grad()
    epoch_loss = 0.0

    for _ in range(num_samples):
        # Generate a random input (for instance, shape [1,1,3,4])
        x = torch.randn(10, 1, 3, 4).cuda()
        
        # Compute target output using the GT kernel.
        with torch.no_grad():
            y_target = Dilation2DFunction.apply(x, w_gt, origin, padding_value)
        
        # Compute prediction from the model.
        y_pred = model(x)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        epoch_loss += loss.item()
    
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {epoch_loss/num_samples:.6f}")

# --- Check Learned Kernel ---
print("\nGround Truth Kernel:")
print(w_gt)
print("\nLearned Kernel:")
print(model.weight.data)
print("\nDifference:")
print(w_gt - model.weight.data)
print("\nL1 Norm of Difference:")
print(torch.norm(w_gt - model.weight.data, p=1))
