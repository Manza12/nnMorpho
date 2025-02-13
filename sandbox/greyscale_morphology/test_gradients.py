import torch
from greyscale_morphology import Dilation2DFunction

device = 'cpu'

# Use small tensors to avoid memory explosion!
x = torch.tensor([[[[3, 2, 0, 1],
                    [1, 2, 1, 0],
                    [1, 2, 3, 1]]]], dtype=torch.float, requires_grad=True, device=device)
w = torch.tensor([[[[0, 2, 1],
                    [1, 0, 3]]]], dtype=torch.float, requires_grad=True, device=device)

# Define functions that return the output y from our custom op.
# Note: We fix one input while computing the Jacobian with respect to the other.
def f_x(x_input):
    # w is fixed
    y = Dilation2DFunction.apply(x_input, w, (1, 1), -float('inf'))
    return y

def f_w(w_input):
    # x is fixed
    y = Dilation2DFunction.apply(x, w_input, (1, 1), -float('inf'))
    return y

# Compute the full Jacobian:
# The resulting jacobian has shape: input.shape + output.shape.
jacobian_x = torch.autograd.functional.jacobian(f_x, x, create_graph=False)
jacobian_w = torch.autograd.functional.jacobian(f_w, w, create_graph=False)

# print("Jacobian of y with respect to x has shape:", jacobian_x.shape)
print("Jacobian of y with respect to w has shape:", jacobian_w.shape)


# print("Jacobian of y with respect to x:", jacobian_x)
print("Jacobian of y with respect to w:", jacobian_w)
