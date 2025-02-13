#include <torch/extension.h>
#include <vector>
#include <limits>

// -----------------------------------
// Forward kernel
// -----------------------------------
__global__ void morphological_dilation2d_forward_kernel(
    const float* __restrict__ input,  // [N, Cin, Hin, Win]
    const float* __restrict__ weight, // [Cout, Cin, Kh, Kw]
    float* __restrict__ out,          // [N, Cout, Hout, Wout]
    int* __restrict__ argmax,         // [N, Cout, Hout, Wout]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int Hout, int Wout,
    int originH, int originW,
    float padding_value
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (tid >= total) return;

    int wo = tid % Wout;
    int tmp = tid / Wout;
    int ho = tmp % Hout;
    tmp /= Hout;
    int co = tmp % Cout;
    int n  = tmp / Cout;

    float best_val = padding_value;
    int best_idx = -1;

    for (int ci = 0; ci < Cin; ci++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw = 0; kw < Kw; kw++) {
                int hi = ho + originH - kh;
                int wi = wo + originW - kw;

                if (hi >= 0 && hi < Hin && wi >= 0 && wi < Win) {
                    float val = input[((n * Cin + ci) * Hin + hi) * Win + wi] +
                                weight[((co * Cin + ci) * Kh + kh) * Kw + kw];
                    if (val > best_val) {
                        best_val = val;
                        best_idx = ci * (Kh * Kw) + kh * Kw + kw;
                    }
                }
            }
        }
    }

    out[tid] = best_val;
    argmax[tid] = best_idx;
}

// -----------------------------------
// Backward kernel
// -----------------------------------
__global__ void morphological_dilation2d_backward_kernel(
    const float* __restrict__ grad_out, // [N, Cout, Hout, Wout]
    const int* __restrict__ argmax,     // [N, Cout, Hout, Wout]
    float* __restrict__ grad_in,        // [N, Cin, Hin, Win]
    float* __restrict__ grad_w,         // [Cout, Cin, Kh, Kw]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int Hout, int Wout,
    int originH, int originW
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (tid >= total) return;

    float go = grad_out[tid];
    int idx = argmax[tid];

    if (idx == -1) return;

    int wo = tid % Wout;
    int tmp = tid / Wout;
    int ho = tmp % Hout;
    tmp /= Hout;
    int co = tmp % Cout;
    int n  = tmp / Cout;

    int ci  = idx / (Kh * Kw);
    int r   = idx % (Kh * Kw);
    int kh  = r / Kw;
    int kw  = r % Kw;

    int hi = ho + originH - kh;
    int wi = wo + originW - kw;


    if (hi >= 0 && hi < Hin && wi >= 0 && wi < Win) {
        atomicAdd(&grad_in[((n * Cin + ci) * Hin + hi) * Win + wi], go);
        atomicAdd(&grad_w[((co * Cin + ci) * Kh + kh) * Kw + kw], go);
    }
}

// -----------------------------------
// Wrappers
// -----------------------------------
std::vector<torch::Tensor> morphological_dilation2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    int N = input.size(0), Cin = input.size(1), Hin = input.size(2), Win = input.size(3);
    int Cout = weight.size(0), Kh = weight.size(2), Kw = weight.size(3);
    int Hout = Hin, Wout = Win;

    auto out = torch::empty({N, Cout, Hout, Wout}, input.options());
    auto argmax = torch::empty({N, Cout, Hout, Wout},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

    int total = N * Cout * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    morphological_dilation2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(),
        argmax.data_ptr<int>(), N, Cin, Hin, Win, Cout, Kh, Kw, Hout, Wout, originH, originW, padding_value);

    return {out, argmax};
}


std::vector<torch::Tensor> morphological_dilation2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int originH,
    int originW,
    float padding_value
) {
    int N = input.size(0), Cin = input.size(1), Hin = input.size(2), Win = input.size(3);
    int Cout = weight.size(0), Kh = weight.size(2), Kw = weight.size(3);
    int Hout = Hin, Wout = Win;

    auto grad_in = torch::zeros_like(input);
    auto grad_w = torch::zeros_like(weight);

    int total = N * Cout * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    morphological_dilation2d_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        argmax.data_ptr<int>(),
        grad_in.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, Kh, Kw,
        Hout, Wout,
        originH, originW
    );

    return {grad_in, grad_w};
}
