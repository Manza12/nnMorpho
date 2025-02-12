import torch
import morphological_dilation2d

# Example usage (CPU)
x = torch.randn(2, 3, 5, 5)  # [N,C,H,W]
w = torch.randn(4, 3, 3, 3)  # [Cout, Cin, Kh, Kw]

out, argm = morphological_dilation2d.morphological_dilation2d_forward(x, w)
grad_out = torch.randn_like(out)
grads = morphological_dilation2d.morphological_dilation2d_backward(grad_out, argm, x, w)
grad_x, grad_w = grads
print("out:", out.shape, "argmax:", argm.shape)
print("grad_x:", grad_x.shape, "grad_w:", grad_w.shape)
