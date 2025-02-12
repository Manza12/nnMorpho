#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void linear_forward_kernel(float* input, float* weight, float* bias, float* output,
                                      int batch_size, int in_features, int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * out_features) {
        int row = i / out_features;
        int col = i % out_features;
        float sum = bias[col];
        for (int j = 0; j < in_features; j++) {
            sum += input[row * in_features + j] * weight[j * out_features + col];
        }
        output[i] = fmaxf(sum, 0.0f); // ReLU activation
    }
}

torch::Tensor mlp_forward_cuda(torch::Tensor input, torch::Tensor weights1, torch::Tensor bias1,
                               torch::Tensor weights2, torch::Tensor bias2) {
    auto output1 = torch::empty({input.size(0), weights1.size(1)}, input.options());
    auto output2 = torch::empty({input.size(0), weights2.size(1)}, input.options());

    int threads = 256;
    int blocks = (output1.numel() + threads - 1) / threads;

    linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weights1.data_ptr<float>(), bias1.data_ptr<float>(),
        output1.data_ptr<float>(), input.size(0), input.size(1), weights1.size(1));

    blocks = (output2.numel() + threads - 1) / threads;
    linear_forward_kernel<<<blocks, threads>>>(
        output1.data_ptr<float>(), weights2.data_ptr<float>(), bias2.data_ptr<float>(),
        output2.data_ptr<float>(), output1.size(0), output1.size(1), weights2.size(1));

    return output2;
}
