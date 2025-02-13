import torch

def dilation2d_forward(input, weight, origin=None, padding_value=-float('inf')):
    """
    Perform 2D morphological dilation on the input tensor.

    Args:
        input (torch.Tensor): Input tensor of shape [N, Cin, H, W].
        weight (torch.Tensor): Weight tensor of shape [Cout, Cin, Kh, Kw].
        origin (tuple): (originH, originW) for the kernel.
        padding_value (float): Value used for padding (e.g., -inf for max).
    
    Returns:
        out (torch.Tensor): Output tensor of shape [N, Cout, H, W].
        argmax (torch.Tensor): Tensor of indices (same shape as out) recording
                               the location of the max value (for backward use).
    """
    N, Cin, H, W = input.shape
    Cout, WCin, Kh, Kw = weight.shape
    if Cin != WCin:
        raise ValueError("Input channel count must match weight's in-channel count.")
    
    # Output will have the same spatial dimensions as input.
    out = torch.full((N, Cout, H, W), padding_value, dtype=input.dtype, device=input.device)
    argmax = torch.full((N, Cout, H, W), -1, dtype=torch.int32, device=input.device)

    if origin is None:
        origin = (Kh // 2, Kw // 2)
    originH, originW = origin

    # Loop over each element in the output tensor.
    for n in range(N):
        for co in range(Cout):
            for ho in range(H):
                for wo in range(W):
                    best_val = padding_value
                    best_idx = -1
                    # Loop over input channels and kernel elements.
                    for ci in range(Cin):
                        for kh in range(Kh):
                            for kw in range(Kw):
                                hi = ho + originH - kh
                                wi = wo + originW - kw
                                if 0 <= hi < H and 0 <= wi < W:
                                    candidate = input[n, ci, hi, wi] + weight[co, ci, kh, kw]
                                    if candidate > best_val:
                                        best_val = candidate
                                        # Store the index as in the C++ version:
                                        best_idx = ci * (Kh * Kw) + kh * Kw + kw
                    out[n, co, ho, wo] = best_val
                    argmax[n, co, ho, wo] = best_idx

    return out, argmax

if __name__ == "__main__":
    # Define input and weight tensors as in your example.
    x = torch.tensor([[[[3, 2, 0, 1],
                        [1, 2, 1, 0],
                        [1, 2, 3, 1]]]]).float()
    w = torch.tensor([[[[0, 2, 1],
                        [1, 0, 3]]]]).float()

    origin = (1, 1)
    padding_value = -float('inf')

    # Compute the dilation.
    y, argmax = dilation2d_forward(x, w, origin, padding_value)

    print("x =", x)
    print("w =", w)
    print("y =", y)
