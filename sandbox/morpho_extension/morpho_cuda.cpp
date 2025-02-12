#include <torch/extension.h>
#include <vector>

// Forward declarations of our CUDA kernels
std::vector<torch::Tensor> morphological_forward_cuda(
    torch::Tensor inp,
    torch::Tensor weight
);

std::vector<torch::Tensor> morphological_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor inp,
    torch::Tensor weight,
    torch::Tensor argmax
);

// -----------------------------------------------------------------------------
// C++ "wrapper" functions exposed to Python
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> morphological_forward(
    torch::Tensor inp,
    torch::Tensor weight
) {
    // Will return (out, argmax)
    return morphological_forward_cuda(inp, weight);
}

std::vector<torch::Tensor> morphological_backward(
    torch::Tensor grad_out,
    torch::Tensor inp,
    torch::Tensor weight,
    torch::Tensor argmax
) {
    // Will return (grad_inp, grad_weight)
    return morphological_backward_cuda(grad_out, inp, weight, argmax);
}

// -----------------------------------------------------------------------------
// PyBind module definition
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morphological_forward",
          &morphological_forward,
          "Single-layer morphological forward (no bias)");
    m.def("morphological_backward",
          &morphological_backward,
          "Single-layer morphological backward (no bias)");
}
