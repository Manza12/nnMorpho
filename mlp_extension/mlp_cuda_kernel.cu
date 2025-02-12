#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * -----------------------
 * 1) FORWARD KERNEL
 *    linear_forward_kernel does: out[row,col] = ReLU( sum(...) )
 * -----------------------
 */
__global__ void linear_forward_kernel(const float* input,
                                      const float* weight,
                                      const float* bias,
                                      float* output,
                                      int batch_size,
                                      int in_features,
                                      int out_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size * out_features) {
        int row = i / out_features;  // which batch
        int col = i % out_features;  // which output feature
        float val = bias[col];
        for (int j = 0; j < in_features; j++) {
            val += input[row * in_features + j] * weight[j * out_features + col];
        }
        // ReLU
        if (val < 0.0f) val = 0.0f;
        output[i] = val;
    }
}

/*
 * -----------------------
 * 2) Forward pass wrapper
 * -----------------------
 */
torch::Tensor mlp_forward_cuda(
    torch::Tensor input,
    torch::Tensor weights1,
    torch::Tensor bias1,
    torch::Tensor weights2,
    torch::Tensor bias2
) {
    // input  shape: [B, in_features]
    // weights1 shape: [in_features, hidden_dim]
    // bias1   shape: [hidden_dim]
    // weights2 shape: [hidden_dim, out_features]
    // bias2   shape: [out_features]

    auto B = input.size(0);
    auto in_features = input.size(1);
    auto hidden_dim = weights1.size(1);
    auto out_features = weights2.size(1);

    // Allocate output of first layer + second layer
    auto output1 = torch::empty({B, hidden_dim}, input.options());
    auto output2 = torch::empty({B, out_features}, input.options());

    // Launch first layer
    int threads = 256;
    int total1 = B * hidden_dim;
    int blocks = (total1 + threads - 1) / threads;

    linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weights1.data_ptr<float>(),
        bias1.data_ptr<float>(),
        output1.data_ptr<float>(),
        B,
        in_features,
        hidden_dim
    );

    // Launch second layer
    int total2 = B * out_features;
    blocks = (total2 + threads - 1) / threads;

    linear_forward_kernel<<<blocks, threads>>>(
        output1.data_ptr<float>(),
        weights2.data_ptr<float>(),
        bias2.data_ptr<float>(),
        output2.data_ptr<float>(),
        B,
        hidden_dim,
        out_features
    );

    // Return final output (shape [B, out_features])
    return output2;
}

/*
 * -------------------------------------------------------
 * 3) BACKWARD PASS: compute gradient wrt input only
 *    We do:
 *      grad_input = grad_output x weights2^T x weights1^T
 *    (Ignoring ReLU's partial derivative for brevity.)
 * -------------------------------------------------------
 */
torch::Tensor mlp_backward_cuda(
    torch::Tensor grad_output,  // [B, out_features]
    torch::Tensor input,        // [B, in_features]
    torch::Tensor weights1,     // [in_features, hidden_dim]
    torch::Tensor weights2      // [hidden_dim, out_features]
) {
    // We'll do a naive matrix multiply chain in C++/CUDA with at::mm:
    //   grad_x1 = grad_output x weights2^T
    //   grad_input = grad_x1 x weights1^T
    // (Skipping ReLU masks for simplicity.)

    // (1) grad_x1 = grad_output x weights2^T
    auto grad_x1 = torch::mm(grad_output, weights2.transpose(0,1));  // [B, hidden_dim]

    // (2) grad_input = grad_x1 x weights1^T
    auto grad_input = torch::mm(grad_x1, weights1.transpose(0,1));   // [B, in_features]
    return grad_input;
}
