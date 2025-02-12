#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA functions
torch::Tensor mlp_forward_cuda(torch::Tensor input, torch::Tensor weights1, torch::Tensor bias1,
                               torch::Tensor weights2, torch::Tensor bias2);

torch::Tensor mlp_backward_cuda(torch::Tensor grad_output, torch::Tensor input,
                                torch::Tensor weights1, torch::Tensor weights2);

// Wrapper function for Python
torch::Tensor mlp_forward(torch::Tensor input, torch::Tensor weights1, torch::Tensor bias1,
                          torch::Tensor weights2, torch::Tensor bias2) {
    return mlp_forward_cuda(input, weights1, bias1, weights2, bias2);
}

torch::Tensor mlp_backward(torch::Tensor grad_output, torch::Tensor input,
                           torch::Tensor weights1, torch::Tensor weights2) {
    return mlp_backward_cuda(grad_output, input, weights1, weights2);
}

// Define module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mlp_forward, "MLP forward pass");
    m.def("backward", &mlp_backward, "MLP backward pass");
}
