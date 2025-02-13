import torch
from greyscale_morphology import Dilation2DFunction

device = 'cuda'

x = torch.tensor([[[[3, 2, 0, 1], [1, 2, 1, 0], [1, 2, 3, 1]]]]).float()
w = torch.tensor([[[[0, 2, 1], [1, 0, 3]]]]).float()

if device == 'cuda':
    x = x.cuda()
    w = w.cuda()

print("x =", x)
print("w =", w)

y = Dilation2DFunction.apply(x, w, (1, 1), -float('inf'))

print("y =", y)
