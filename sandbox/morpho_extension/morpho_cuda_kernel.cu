#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

///////////////////////////////////////////////////
// 1) FORWARD KERNEL: Single-layer morphological
///////////////////////////////////////////////////
__global__ void morphological_forward_kernel(
    const float* __restrict__ inp,     // [B, in_features]
    const float* __restrict__ weight,  // [in_features, out_features]
    float* __restrict__ out,           // [B, out_features]
    int* __restrict__ argmax,          // [B, out_features]
    int B,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * out_features) {
        int b = idx / out_features;
        int j = idx % out_features;

        float best_val = -std::numeric_limits<float>::infinity();
        int best_idx = -1;

        // Find max_i( inp[b,i] + weight[i,j] )
        for (int i = 0; i < in_features; i++) {
            float val = inp[b * in_features + i] + weight[i * out_features + j];
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        }
        out[b * out_features + j] = best_val;
        argmax[b * out_features + j] = best_idx;
    }
}

///////////////////////////////////////////////////
// 2) morphological_forward_cuda: returns (out, argmax)
///////////////////////////////////////////////////
std::vector<torch::Tensor> morphological_forward_cuda(
    torch::Tensor inp,      // [B, in_features]
    torch::Tensor weight    // [in_features, out_features]
) {
    int B = inp.size(0);
    int in_features = inp.size(1);
    int out_features = weight.size(1);

    // Allocate output
    auto out = torch::empty({B, out_features}, inp.options());
    auto argmax = torch::empty({B, out_features},
                     torch::TensorOptions().dtype(torch::kInt32).device(inp.device()));

    // Launch
    int threads = 256;
    int total = B * out_features;
    int blocks = (total + threads - 1) / threads;

    morphological_forward_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        argmax.data_ptr<int>(),
        B,
        in_features,
        out_features
    );

    return {out, argmax};
}


///////////////////////////////////////////////////
// 3) BACKWARD KERNEL: Single-layer morphological
///////////////////////////////////////////////////
__global__ void morphological_backward_kernel(
    const float* __restrict__ grad_out,  // [B, out_features]
    const int* __restrict__ argmax,      // [B, out_features]
    float* __restrict__ grad_inp,        // [B, in_features]
    float* __restrict__ grad_weight,     // [in_features, out_features]
    int B,
    int in_features,
    int out_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * out_features) {
        int b = idx / out_features;
        int j = idx % out_features;

        float go = grad_out[b * out_features + j];   // gradient w.r.t out[b,j]
        int i_star = argmax[b * out_features + j];   // winning index

        // grad_inp[b, i_star] += go
        atomicAdd(&grad_inp[b * in_features + i_star], go);

        // grad_weight[i_star, j] += go
        atomicAdd(&grad_weight[i_star * out_features + j], go);
    }
}

///////////////////////////////////////////////////
// 4) morphological_backward_cuda: returns (grad_inp, grad_weight)
///////////////////////////////////////////////////
std::vector<torch::Tensor> morphological_backward_cuda(
    torch::Tensor grad_out,   // [B, out_features]
    torch::Tensor inp,        // [B, in_features]
    torch::Tensor weight,     // [in_features, out_features]
    torch::Tensor argmax      // [B, out_features]
) {
    int B = inp.size(0);
    int in_features = inp.size(1);
    int out_features = weight.size(1);

    // Create gradients
    auto grad_inp = torch::zeros_like(inp);
    auto grad_weight = torch::zeros_like(weight);

    // Launch
    int threads = 256;
    int total = B * out_features;
    int blocks = (total + threads - 1) / threads;

    morphological_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        argmax.data_ptr<int>(),
        grad_inp.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        B,
        in_features,
        out_features
    );

    return {grad_inp, grad_weight};
}
