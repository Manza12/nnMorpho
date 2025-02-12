#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <limits>
#include <iostream>

// Declarations from the .cu file
std::vector<torch::Tensor> morphological_dilation2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
);

std::vector<torch::Tensor> morphological_dilation2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
);

// -----------------------------------------------------------------------------
// CPU fallback for 2D morphological dilation (forward + backward).
// No stride/padding/dilation/groups in this minimal example.
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> morphological_dilation2d_forward_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    // Ensure input is on CPU
    TORCH_CHECK(!input.is_cuda(), "Input must be a CPU tensor.");
    TORCH_CHECK(!weight.is_cuda(), "Weight must be a CPU tensor.");

    auto N = input.size(0);
    auto Cin = input.size(1);
    auto Hin = input.size(2);
    auto Win = input.size(3);

    auto Cout = weight.size(0);
    auto WCin = weight.size(1);
    auto Kh = weight.size(2);
    auto Kw = weight.size(3);

    TORCH_CHECK(Cin == WCin, "Input channel count must match weight's in-channel count.");

    // Compute output dimensions
    int Hout = Hin;
    int Wout = Win;

    // Allocate output tensor
    auto out = torch::full({N, Cout, Hout, Wout}, padding_value, input.options());
    auto argmax = torch::empty({N, Cout, Hout, Wout}, torch::TensorOptions().dtype(torch::kInt32));

    // Pointers for fast access
    float* inp_ptr = input.data_ptr<float>();
    float* w_ptr = weight.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    int* argmax_ptr = argmax.data_ptr<int>();

    // Perform dilation
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < Cout; co++) {
            for (int ho = 0; ho < Hout; ho++) {
                for (int wo = 0; wo < Wout; wo++) {

                    float best_val = padding_value;
                    int best_idx = -1;

                    for (int ci = 0; ci < Cin; ci++) {
                        for (int kh = 0; kh < Kh; kh++) {
                            for (int kw = 0; kw < Kw; kw++) {
                                int hi = ho + kh - originH;
                                int wi = wo + kw - originW;

                                if (hi >= 0 && hi < Hin && wi >= 0 && wi < Win) {
                                    float val = inp_ptr[((n * Cin + ci) * Hin + hi) * Win + wi]
                                                + w_ptr[((co * Cin + ci) * Kh + kh) * Kw + kw];

                                    if (val > best_val) {
                                        best_val = val;
                                        best_idx = ci * (Kh * Kw) + kh * Kw + kw;
                                    }
                                }
                            }
                        }
                    }

                    out_ptr[((n * Cout + co) * Hout + ho) * Wout + wo] = best_val;
                    argmax_ptr[((n * Cout + co) * Hout + ho) * Wout + wo] = best_idx;
                }
            }
        }
    }

    return {out, argmax};
}

std::vector<torch::Tensor> morphological_dilation2d_backward_cpu(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    // Ensure CPU tensors
    TORCH_CHECK(!grad_out.is_cuda(), "grad_out must be a CPU tensor.");
    TORCH_CHECK(!argmax.is_cuda(), "argmax must be a CPU tensor.");

    auto N = input.size(0);
    auto Cin = input.size(1);
    auto Hin = input.size(2);
    auto Win = input.size(3);

    auto Cout = weight.size(0);
    auto WCin = weight.size(1);
    auto Kh = weight.size(2);
    auto Kw = weight.size(3);

    TORCH_CHECK(Cin == WCin, "Input channel count must match weight's in-channel count.");

    // Compute output dimensions
    int Hout = Hin;
    int Wout = Win;

    // Allocate gradient tensors
    auto grad_in = torch::zeros_like(input);
    auto grad_w = torch::zeros_like(weight);

    float* g_out_ptr = grad_out.data_ptr<float>();
    int* argmax_ptr = argmax.data_ptr<int>();
    float* g_in_ptr = grad_in.data_ptr<float>();
    float* g_w_ptr = grad_w.data_ptr<float>();

    // Backward pass computation
    for (int n = 0; n < N; n++) {
        for (int co = 0; co < Cout; co++) {
            for (int ho = 0; ho < Hout; ho++) {
                for (int wo = 0; wo < Wout; wo++) {
                    float go = g_out_ptr[((n * Cout + co) * Hout + ho) * Wout + wo];
                    int idx = argmax_ptr[((n * Cout + co) * Hout + ho) * Wout + wo];

                    if (idx != -1) {
                        int ci = idx / (Kh * Kw);
                        int rem = idx % (Kh * Kw);
                        int kh = rem / Kw;
                        int kw = rem % Kw;

                        int hi = ho + kh - originH;
                        int wi = wo + kw - originW;

                        if (hi >= 0 && hi < Hin && wi >= 0 && wi < Win) {
                            g_in_ptr[((n * Cin + ci) * Hin + hi) * Win + wi] += go;
                            g_w_ptr[((co * Cin + ci) * Kh + kh) * Kw + kw] += go;
                        }
                    }
                }
            }
        }
    }

    return {grad_in, grad_w};
}

// -----------------------------------------------------------------------------
// CUDA Dispatch
// -----------------------------------------------------------------------------

std::vector<torch::Tensor> morphological_dilation2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    if (input.is_cuda()) {
        return morphological_dilation2d_forward_cuda(input, weight, originH, originW, padding_value);
    } else {
        return morphological_dilation2d_forward_cpu(input, weight, originH, originW, padding_value);
    }
}

std::vector<torch::Tensor> morphological_dilation2d_backward(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    if (grad_out.is_cuda()) {
        return morphological_dilation2d_backward_cuda(grad_out, argmax, input, weight, originH, originW, padding_value);
    } else {
        return morphological_dilation2d_backward_cpu(grad_out, argmax, input, weight, originH, originW, padding_value);
    }
}

// -----------------------------------------------------------------------------
// PYBIND11 Module definition
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morphological_dilation2d_forward", &morphological_dilation2d_forward,
          "2D morphological dilation forward (CUDA)");
    m.def("morphological_dilation2d_backward", &morphological_dilation2d_backward,
          "2D morphological dilation backward (CUDA)");
}
