#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>

// Declarations from the .cpp file
std::vector<torch::Tensor> dilation2d_forward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
);

std::vector<torch::Tensor> dilation2d_backward_cpu(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
);

// Declarations from the .cu file
std::vector<torch::Tensor> dilation2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
);

std::vector<torch::Tensor> dilation2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
);

// -----------------------------------------------------------------------------
// C++/CUDA Dispatch
// -----------------------------------------------------------------------------

std::vector<torch::Tensor> dilation2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    if (input.is_cuda()) {
        return dilation2d_forward_cuda(input, weight, originH, originW, padding_value);
    } else {
        return dilation2d_forward_cpu(input, weight, originH, originW, padding_value);
    }
}

std::vector<torch::Tensor> dilation2d_backward(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    if (grad_out.is_cuda()) {
        return dilation2d_backward_cuda(grad_out, argmax, input, weight, originH, originW, padding_value);
    } else {
        return dilation2d_backward_cpu(grad_out, argmax, input, weight, originH, originW, padding_value);
    }
}

// -----------------------------------------------------------------------------
// PYBIND11 Module definition
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dilation2d_forward", &dilation2d_forward,
          "2D morphological dilation forward");
    m.def("dilation2d_backward", &dilation2d_backward,
          "2D morphological dilation backward");
}
