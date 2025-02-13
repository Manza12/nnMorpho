import torch
from modules import Dilation2d, Dilation2dFunction

device = 'cpu'
layer = Dilation2d(in_channels=1, out_channels=1, kernel_size=3, origin=1, padding_value=-float('inf'))
if device == 'cuda':
    layer = layer.cuda()

x = torch.tensor([[[[3, 2, 0, 1], [1, 2, 1, 0], [1, 2, 3, 1]]]]).float()
w = torch.tensor([[[[0, 2, 1], [1, 0, 3]]]]).float()

if device == 'cuda':
    x = x.cuda()
    w = w.cuda()

print("x =", x)
print("w =", w)

out = layer(x)
y = Dilation2dFunction.apply(x, w, (1, 0), -float('inf'))

# print("Output shape:", out.shape)
print("y =", y)
