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
    int padH, int padW,
    bool useNegInfPad
)
{
    // Each thread -> one output element (n, co, ho, wo)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (tid >= total) return;

    // decode (n, co, ho, wo)
    int wo = tid % Wout;
    int tmp = tid / Wout;
    int ho = tmp % Hout;
    tmp = tmp / Hout;
    int co = tmp % Cout;
    int n  = tmp / Cout;

    float best_val = -std::numeric_limits<float>::infinity();
    int best_idx = -1;

    // For each (ci, kh, kw_)
    for (int ci = 0; ci < Cin; ci++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw_ = 0; kw_ < Kw; kw_++) {
                // re-map from output (ho, wo) -> input coords
                int hi = (ho - padH) + kh;
                int wi = (wo - padW) + kw_;

                float val;
                bool inBounds = (hi >= 0 && hi < Hin && wi >= 0 && wi < Win);

                if (inBounds) {
                    float inp_val = input[((n * Cin + ci)*Hin + hi)*Win + wi];
                    float w_val   = weight[((co * Cin + ci)*Kh + kh)*Kw + kw_];
                    val = inp_val + w_val;
                } else {
                    // out-of-bounds => either -∞ or 0
                    val = (useNegInfPad ?
                           -std::numeric_limits<float>::infinity() :
                            0.0f);
                }

                if (val > best_val) {
                    best_val = val;
                    // flatten (ci, kh, kw_)
                    best_idx = ci*(Kh*Kw) + kh*Kw + kw_;
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
    const int*  __restrict__ argmax,    // [N, Cout, Hout, Wout]
    float* __restrict__ grad_in,        // [N, Cin, Hin, Win]
    float* __restrict__ grad_w,         // [Cout, Cin, Kh, Kw]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int Hout, int Wout,
    int padH, int padW,
    bool useNegInfPad
)
{
    // Each thread -> one output element (n, co, ho, wo)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (tid >= total) return;

    float go = grad_out[tid];   // gradient wrt out[n, co, ho, wo]
    int idx  = argmax[tid];     // which (ci, kh, kw_) was max?

    // decode (n, co, ho, wo)
    int wo = tid % Wout;
    int tmp = tid / Wout;
    int ho = tmp % Hout;
    tmp = tmp / Hout;
    int co = tmp % Cout;
    int n  = tmp / Cout;

    // decode (ci, kh, kw_)
    int ci  = idx / (Kh * Kw);
    int r   = idx % (Kh * Kw);
    int kh  = r / Kw;
    int kw_ = r % Kw;

    // re-map to input coords
    int hi = (ho - padH) + kh;
    int wi = (wo - padW) + kw_;

    // skip if out-of-bounds
    if (hi < 0 || hi >= Hin || wi < 0 || wi >= Win) {
        return;
    }

    // atomicAdd to grad_in, grad_w
    atomicAdd(&grad_in[((n * Cin + ci)*Hin + hi)*Win + wi], go);
    atomicAdd(&grad_w[((co * Cin + ci)*Kh + kh)*Kw + kw_], go);
}

// -----------------------------------
// Wrappers
// -----------------------------------
std::vector<torch::Tensor> morphological_dilation2d_forward_cuda(
    torch::Tensor input,    // [N, Cin, Hin, Win]
    torch::Tensor weight,   // [Cout, Cin, Kh, Kw]
    int padH,
    int padW,
    bool useNegInfPad
) {
    auto N    = input.size(0);
    auto Cin  = input.size(1);
    auto Hin  = input.size(2);
    auto Win  = input.size(3);

    auto Cout = weight.size(0);
    auto WCin = weight.size(1);
    auto Kh   = weight.size(2);
    auto Kw   = weight.size(3);

    // compute output size
    int Hout = Hin + 2*padH - Kh + 1;  // stride=1
    int Wout = Win + 2*padW - Kw + 1;

    auto out = torch::empty({N, Cout, Hout, Wout}, input.options());
    auto argmax = torch::empty({N, Cout, Hout, Wout},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

    int total = N * Cout * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // Launch
    morphological_dilation2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        argmax.data_ptr<int>(),
        N, Cin, Hin, Win,
        Cout, Kh, Kw,
        Hout, Wout,
        padH, padW,
        useNegInfPad
    );

    return {out, argmax};
}

std::vector<torch::Tensor> morphological_dilation2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight,
    int padH,
    int padW,
    bool useNegInfPad
) {
    int N    = input.size(0);
    int Cin  = input.size(1);
    int Hin  = input.size(2);
    int Win  = input.size(3);

    int Cout = weight.size(0);
    int WCin = weight.size(1);
    int Kh   = weight.size(2);
    int Kw   = weight.size(3);

    // same shape logic as forward
    int Hout = Hin + 2*padH - Kh + 1;
    int Wout = Win + 2*padW - Kw + 1;

    auto grad_in = torch::zeros_like(input);
    auto grad_w  = torch::zeros_like(weight);

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
        padH, padW,
        useNegInfPad
    );

    return {grad_in, grad_w};
}
