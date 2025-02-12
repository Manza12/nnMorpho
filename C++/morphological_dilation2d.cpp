#include <torch/extension.h>
#include <vector>
#include <limits>
#include <iostream>

// -----------------------------------------------------------------------------
// CPU fallback for 2D morphological dilation (forward + backward).
// No stride/padding/dilation/groups in this minimal example.
// -----------------------------------------------------------------------------

// Forward pass:
//   input:  [N, Cin, Hin, Win]
//   weight: [Cout, Cin, Kh, Kw]
// Output shape: [N, Cout, Hout, Wout]
//   where Hout = Hin - Kh + 1, Wout = Win - Kw + 1  (no padding, stride=1)
// Argmax: same shape as output (stores index in [Cin, Kh, Kw], flattened).
std::vector<torch::Tensor> morphological_dilation2d_forward_cpu(
    torch::Tensor input,
    torch::Tensor weight
) {
    // Check device
    TORCH_CHECK(!input.is_cuda(), "input must be CPU tensor");
    TORCH_CHECK(!weight.is_cuda(), "weight must be CPU tensor");

    auto N     = input.size(0);
    auto Cin   = input.size(1);
    auto Hin   = input.size(2);
    auto Win   = input.size(3);

    auto Cout  = weight.size(0);
    auto WCin  = weight.size(1);
    auto Kh    = weight.size(2);
    auto Kw    = weight.size(3);

    TORCH_CHECK(Cin == WCin, "input channel dim must match weight's in-channel dim");

    // Compute output spatial size (naive, no padding)
    int Hout = Hin - Kh + 1;
    int Wout = Win - Kw + 1;
    TORCH_CHECK(Hout >= 1 && Wout >= 1,
        "Kernel too large for input size (no padding).");

    // Allocate output + argmax
    auto out = torch::empty({N, Cout, Hout, Wout}, input.options());
    // We'll store a single int that encodes which (c_in, kh, kw) gave the max
    auto argmax = torch::empty({N, Cout, Hout, Wout},
                               torch::TensorOptions().dtype(torch::kInt32));

    // Pointers for faster access
    float* inp_ptr    = input.data_ptr<float>();
    float* w_ptr      = weight.data_ptr<float>();
    float* out_ptr    = out.data_ptr<float>();
    int*   argmax_ptr = argmax.data_ptr<int>();

    // Naive nested loops
    for (int n = 0; n < N; n++) {
      for (int co = 0; co < Cout; co++) {
        for (int ho = 0; ho < Hout; ho++) {
          for (int wo = 0; wo < Wout; wo++) {

            float best_val = -std::numeric_limits<float>::infinity();
            int best_idx = -1;

            // Search over Cin, Kh, Kw
            for (int ci = 0; ci < Cin; ci++) {
              for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                  int hi = ho + kh; // no stride/padding
                  int wi = wo + kw;
                  float val = inp_ptr[ ((n * Cin + ci) * Hin + hi) * Win + wi ]
                              + w_ptr[ ((co * Cin + ci) * Kh + kh) * Kw + kw ];
                  if (val > best_val) {
                    best_val = val;
                    // Flatten (ci, kh, kw) into a single integer
                    best_idx = ci * (Kh * Kw) + kh * Kw + kw;
                  }
                }
              }
            }

            // Store
            out_ptr[ ((n * Cout + co) * Hout + ho) * Wout + wo ] = best_val;
            argmax_ptr[ ((n * Cout + co) * Hout + ho) * Wout + wo ] = best_idx;
          }
        }
      }
    }

    return {out, argmax};
}


// Backward pass:
//   grad_out:  [N, Cout, Hout, Wout]
//   argmax:    [N, Cout, Hout, Wout] (encodes which (ci,kh,kw) gave the max)
//   input:     [N, Cin, Hin, Win]  (only used for shape checking)
//   weight:    [Cout, Cin, Kh, Kw] (only used for shape checking)
// Returns:
//   grad_in:   same shape as input
//   grad_w:    same shape as weight
std::vector<torch::Tensor> morphological_dilation2d_backward_cpu(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight
) {
    // Check CPU
    TORCH_CHECK(!grad_out.is_cuda(), "grad_out must be CPU tensor");
    TORCH_CHECK(!argmax.is_cuda(),   "argmax must be CPU tensor");

    auto N     = input.size(0);
    auto Cin   = input.size(1);
    auto Hin   = input.size(2);
    auto Win   = input.size(3);

    auto Cout  = weight.size(0);
    auto WCin  = weight.size(1);
    auto Kh    = weight.size(2);
    auto Kw    = weight.size(3);

    TORCH_CHECK(Cin == WCin, "input channel dim must match weight's in-channel dim");

    int Hout = Hin - Kh + 1;
    int Wout = Win - Kw + 1;

    // Allocate grad_in, grad_w
    auto grad_in  = torch::zeros_like(input);
    auto grad_w   = torch::zeros_like(weight);

    float* g_out_ptr = grad_out.data_ptr<float>();
    int*   argm_ptr  = argmax.data_ptr<int>();
    float* g_in_ptr  = grad_in.data_ptr<float>();
    float* g_w_ptr   = grad_w.data_ptr<float>();

    // For each output pixel, add grad_out to the winning (ci, kh, kw).
    for (int n = 0; n < N; n++) {
      for (int co = 0; co < Cout; co++) {
        for (int ho = 0; ho < Hout; ho++) {
          for (int wo = 0; wo < Wout; wo++) {
            float go = g_out_ptr[ ((n*Cout + co)*Hout + ho)*Wout + wo ];
            int idx  = argm_ptr[ ((n*Cout + co)*Hout + ho)*Wout + wo ];

            // decode ci, kh, kw
            int ci = idx / (Kh * Kw);
            int r  = idx % (Kh * Kw);
            int kh = r / Kw;
            int kw_ = r % Kw;

            int hi = ho + kh;
            int wi = wo + kw_;

            // grad_in
            g_in_ptr[ ((n*Cin + ci)*Hin + hi)*Win + wi ] += go;

            // grad_w
            g_w_ptr[ ((co*Cin + ci)*Kh + kh)*Kw + kw_ ] += go;
          }
        }
      }
    }

    return {grad_in, grad_w};
}


// -----------------------------------------------------------------------------
// Public entry points that dispatch CPU vs. CUDA.
// For now, we only have the CPU versions, so we call them directly.
// Later, you can add "if (input.is_cuda()) { ... } else { ... }" logic.
// -----------------------------------------------------------------------------

std::vector<torch::Tensor> morphological_dilation2d_forward(
    torch::Tensor input,
    torch::Tensor weight
) {
    // For now, just call CPU fallback
    return morphological_dilation2d_forward_cpu(input, weight);
}

std::vector<torch::Tensor> morphological_dilation2d_backward(
    torch::Tensor grad_out,
    torch::Tensor argmax,
    torch::Tensor input,
    torch::Tensor weight
) {
    // For now, just call CPU fallback
    return morphological_dilation2d_backward_cpu(grad_out, argmax, input, weight);
}

// -----------------------------------------------------------------------------
// PYBIND11 Module definition
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morphological_dilation2d_forward",
          &morphological_dilation2d_forward,
          "2D morphological dilation forward (CPU fallback)");
    m.def("morphological_dilation2d_backward",
          &morphological_dilation2d_backward,
          "2D morphological dilation backward (CPU fallback)");
}
