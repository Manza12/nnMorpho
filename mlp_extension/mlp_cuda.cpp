#include <torch/extension.h>
#include <vector>

// Forward declarations of the real CUDA routines
//   that handle the multi-layer forward/backward:
std::vector<torch::Tensor> mlp_forward_all_cuda(
    torch::Tensor input,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2
);

std::vector<torch::Tensor> mlp_backward_all_cuda(
    torch::Tensor grad_out,  // grad wrt final output
    torch::Tensor input,
    torch::Tensor out1,
    torch::Tensor out2,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2
);

// -----------------------------------------------------------------------------
// C++ "wrapper" functions that call the CUDA routines
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> mlp_forward_all(
    torch::Tensor input,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2
) {
    return mlp_forward_all_cuda(input, w1, b1, w2, b2);
}

std::vector<torch::Tensor> mlp_backward_all(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor out1,
    torch::Tensor out2,
    torch::Tensor w1,
    torch::Tensor b1,
    torch::Tensor w2,
    torch::Tensor b2
) {
    return mlp_backward_all_cuda(grad_out, input, out1, out2, w1, b1, w2, b2);
}

// -----------------------------------------------------------------------------
// PyBind module definition
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // forward_all(...) => returns [out1, out2]
    m.def("forward_all", &mlp_forward_all, "MLP forward pass (CUDA) - returns 2 layers outputs");

    // backward_all(...) => returns [grad_x, grad_w1, grad_b1, grad_w2, grad_b2]
    m.def("backward_all", &mlp_backward_all, "MLP backward pass (CUDA) - full param grads");
}
