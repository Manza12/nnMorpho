#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
  forward_all:

  We do a 2-layer MLP:
    z1 = (x mm w1) + b1
    out1 = relu(z1)
    z2 = (out1 mm w2) + b2
    out2 = relu(z2)

  Return both out1 and out2 so we can use them in backward.
*/
std::vector<torch::Tensor> mlp_forward_all_cuda(
    torch::Tensor input,  // shape [B, in_features]
    torch::Tensor w1,     // [in_features, hidden_dim]
    torch::Tensor b1,     // [hidden_dim]
    torch::Tensor w2,     // [hidden_dim, out_features]
    torch::Tensor b2      // [out_features]
) {
    auto z1 = torch::addmm(b1, input, w1);  // z1 = x*w1 + b1
    auto out1 = torch::relu(z1);

    auto z2 = torch::addmm(b2, out1, w2);   // z2 = out1*w2 + b2
    auto out2 = torch::relu(z2);

    // Return both intermediate (out1) and final (out2)
    return {out1, out2};
}

/*
  backward_all:

  Suppose the final output was out2.
  We have grad_out = dL/d(out2).
  We compute:

   (1) dZ2 = grad_out * ReLU'(z2)
            but we only have out2 => mask = out2>0 => dZ2 = grad_out*(out2>0)

   (2) grad_w2 = out1^T mm dZ2
   (3) grad_b2 = sum(dZ2 across batch)
   (4) grad_out1 = dZ2 mm w2^T

   (5) dZ1 = grad_out1 * ReLU'(z1)
            but we only have out1 => dZ1 = grad_out1*(out1>0)

   (6) grad_w1 = x^T mm dZ1
   (7) grad_b1 = sum(dZ1 across batch)
   (8) grad_x  = dZ1 mm w1^T

  Return (grad_x, grad_w1, grad_b1, grad_w2, grad_b2).
  All the math can be done with standard PyTorch "ATen" ops in C++.
*/
std::vector<torch::Tensor> mlp_backward_all_cuda(
    torch::Tensor grad_out,  // [B, out_features]
    torch::Tensor input,     // [B, in_features]
    torch::Tensor out1,      // [B, hidden_dim]
    torch::Tensor out2,      // [B, out_features]
    torch::Tensor w1,        // [in_features, hidden_dim]
    torch::Tensor b1,        // [hidden_dim]   (unused except for shape)
    torch::Tensor w2,        // [hidden_dim, out_features]
    torch::Tensor b2         // [out_features] (unused except for shape)
) {
    // ----- LAYER 2 backprop -----

    // mask for ReLU on out2
    auto mask2 = out2 > 0;                            // bool tensor
    auto dZ2 = grad_out * mask2.to(grad_out.dtype()); // [B, out_features]

    // gradient wrt weights2, bias2
    auto grad_w2 = out1.transpose(0,1).mm(dZ2);  // [hidden_dim, out_features]
    auto grad_b2 = dZ2.sum({0});                 // [out_features]

    // gradient wrt out1
    auto grad_out1 = dZ2.mm(w2.transpose(0,1));  // [B, hidden_dim]

    // ----- LAYER 1 backprop -----

    // mask for ReLU on out1
    auto mask1 = out1 > 0;
    auto dZ1 = grad_out1 * mask1.to(grad_out1.dtype()); // [B, hidden_dim]

    // gradient wrt weights1, bias1
    auto grad_w1 = input.transpose(0,1).mm(dZ1); // [in_features, hidden_dim]
    auto grad_b1 = dZ1.sum({0});                 // [hidden_dim]

    // gradient wrt input
    auto grad_x = dZ1.mm(w1.transpose(0,1));     // [B, in_features]

    // Return all in correct order
    return {grad_x, grad_w1, grad_b1, grad_w2, grad_b2};
}
