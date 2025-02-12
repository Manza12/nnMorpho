#include <torch/extension.h>
#include <vector>
#include <limits>

// ----------------------------------------------------------------------
// GPU (CUDA) kernels for 2D morphological dilation
//   + a naive backward using argmax
// ----------------------------------------------------------------------

// -----------------------------------
// Forward kernel
// -----------------------------------
__global__ void morphological_dilation2d_forward_kernel(
    const float* __restrict__ input,  // [N, Cin, Hin, Win]
    const float* __restrict__ weight, // [Cout, Cin, Kh, Kw]
    float* __restrict__ out,          // [N, Cout, Hout, Wout]
    int* __restrict__ argmax,         // [N, Cout, Hout, Wout]
    const int N,
    const int Cin,
    const int Hin,
    const int Win,
    const int Cout,
    const int Kh,
    const int Kw,
    const int Hout,
    const int Wout
)
{
    // Each thread handles one output element: (n, co, ho, wo).
    // We'll compute global index: tid < N*Cout*Hout*Wout.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (tid >= total) {
        return;
    }

    // Decode (n, co, ho, wo) from tid
    int wo = tid % Wout;
    int tmp = tid / Wout;
    int ho = tmp % Hout;
    tmp = tmp / Hout;
    int co = tmp % Cout;
    int n  = tmp / Cout;

    float best_val = -std::numeric_limits<float>::infinity();
    int best_idx = -1;

    // For each (ci, kh, kw)
    //  out[n, co, ho, wo] = max_{ci, kh, kw} (input[n, ci, ho+kh, wo+kw] + weight[co, ci, kh, kw])
    for (int ci = 0; ci < Cin; ci++) {
        for (int kh = 0; kh < Kh; kh++) {
            for (int kw_ = 0; kw_ < Kw; kw_++) {
                int hi = ho + kh;  // no stride/padding in this naive example
                int wi = wo + kw_;

                float inp_val = input[ ((n * Cin + ci) * Hin + hi) * Win + wi ];
                float w_val   = weight[ ((co * Cin + ci) * Kh + kh) * Kw + kw_ ];
                float val = inp_val + w_val;

                if (val > best_val) {
                    best_val = val;
                    // flatten (ci, kh, kw)
                    best_idx = ci * (Kh * Kw) + kh * Kw + kw_;
                }
            }
        }
    }

    // Write out
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
    const int N,
    const int Cin,
    const int Hin,
    const int Win,
    const int Cout,
    const int Kh,
    const int Kw,
    const int Hout,
    const int Wout
)
{
    // Each thread handles one (n, co, ho, wo).
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (tid >= total) {
        return;
    }

    float go = grad_out[tid];  // gradient wrt out[n, co, ho, wo]
    int idx  = argmax[tid];    // which (ci, kh, kw) was max?

    // decode ci, kh, kw from idx
    int ci  = idx / (Kh * Kw);
    int r   = idx % (Kh * Kw);
    int kh  = r / Kw;
    int kw_ = r % Kw;

    // decode n, co, ho, wo
    int wo = tid % Wout;
    int tmp = tid / Wout;
    int ho = tmp % Hout;
    tmp = tmp / Hout;
    int co = tmp % Cout;
    int n  = tmp / Cout;

    // compute the actual input coords
    int hi = ho + kh;
    int wi = wo + kw_;

    // Atomic adds to grad_in & grad_w
    atomicAdd(
        &grad_in[((n * Cin + ci) * Hin + hi) * Win + wi],
        go
    );

    atomicAdd(
        &grad_w[((co * Cin + ci) * Kh + kh) * Kw + kw_],
        go
    );
}


std::vector<torch::Tensor> morphological_dilation2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight
) {
    // shapes
    int N    = input.size(0);
    int Cin  = input.size(1);
    int Hin  = input.size(2);
    int Win  = input.size(3);

    int Cout = weight.size(0);
    int WCin = weight.size(1);
    int Kh   = weight.size(2);
    int Kw   = weight.size(3);

    // naive: no padding => Hout = Hin - Kh + 1, Wout = Win - Kw + 1
    int Hout = Hin - Kh + 1;
    int Wout = Win - Kw + 1;

    // allocate out
    auto out = torch::empty({N, Cout, Hout, Wout}, input.options());
    auto argmax = torch::empty({N, Cout, Hout, Wout},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

    // launch
    int total = N * Cout * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    morphological_dilation2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        argmax.data_ptr<int>(),
        N, Cin, Hin, Win, Cout, Kh, Kw, Hout, Wout
    );

    return {out, argmax};
}


std::vector<torch::Tensor> morphological_dilation2d_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight
) {
    int N    = input.size(0);
    int Cin  = input.size(1);
    int Hin  = input.size(2);
    int Win  = input.size(3);

    int Cout = weight.size(0);
    int WCin = weight.size(1);
    int Kh   = weight.size(2);
    int Kw   = weight.size(3);

    int Hout = Hin - Kh + 1;
    int Wout = Win - Kw + 1;

    auto grad_in = torch::zeros_like(input);
    auto grad_w  = torch::zeros_like(weight);

    // launch
    int total = N * Cout * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    morphological_dilation2d_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        argmax.data_ptr<int>(),
        grad_in.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, Kh, Kw, Hout, Wout
    );

    return {grad_in, grad_w};
}
