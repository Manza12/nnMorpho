#include <torch/extension.h>
#include <vector>

// ----------------------------------------------------------------------------
// Declarations of the actual CUDA implementations (in mlp_cuda_kernel.cu):
// ----------------------------------------------------------------------------
torch::Tensor mlp_forward_cuda(
    torch::Tensor input,     // [B, in_features]
    torch::Tensor weights1,  // [in_features, hidden_dim]
    torch::Tensor bias1,     // [hidden_dim]
    torch::Tensor weights2,  // [hidden_dim, out_features]
    torch::Tensor bias2      // [out_features]
);

torch::Tensor mlp_backward_cuda(
    torch::Tensor grad_output,  // [B, out_features]
    torch::Tensor input,        // [B, in_features]
    torch::Tensor weights1,     // [in_features, hidden_dim]
    torch::Tensor weights2      // [hidden_dim, out_features]
);

// ----------------------------------------------------------------------------
// C++ "wrapper" functions that call the CUDA code
// ----------------------------------------------------------------------------
torch::Tensor mlp_forward(
    torch::Tensor input,
    torch::Tensor weights1,
    torch::Tensor bias1,
    torch::Tensor weights2,
    torch::Tensor bias2
) {
    return mlp_forward_cuda(input, weights1, bias1, weights2, bias2);
}

torch::Tensor mlp_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor weights1,
    torch::Tensor weights2
) {
    return mlp_backward_cuda(grad_output, input, weights1, weights2);
}

// ----------------------------------------------------------------------------
// PyBind module definition
// ----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mlp_forward, "MLP forward pass (CUDA)");
    m.def("backward", &mlp_backward, "MLP backward pass (CUDA, grad wrt input)");
}
