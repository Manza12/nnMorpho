import torch
from modules import Dilation2dFunction

device = 'cpu'

x = torch.tensor([[[[3, 2, 0, 1], [1, 2, 1, 0], [1, 2, 3, 1]]]]).float()
w = torch.tensor([[[[0, 2, 1], [1, 0, 3]]]]).float()

if device == 'cuda':
    x = x.cuda()
    w = w.cuda()

print("x =", x)
print("w =", w)

y = Dilation2dFunction.apply(x, w, (1, 0), -float('inf'))

print("y =", y)
