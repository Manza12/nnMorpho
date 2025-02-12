__global__ void morphological_dilation2d_backward_kernel(
    const float* __restrict__ grad_out, // [N, Cout, Hout, Wout]
    const int*  __restrict__ argmax,    // [N, Cout, Hout, Wout]
    float* __restrict__ grad_in,        // [N, Cin, Hin, Win]
    float* __restrict__ grad_w,         // [Cout, Cin, Kh, Kw]
    int N, int Cin, int Hin, int Win,
    int Cout, int Kh, int Kw,
    int Hout, int Wout,
    int padH, int padW,
    bool useNegInfPad  // Not strictly used if you only skip OOB, but included if needed
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (tid >= total) return;

    float go = grad_out[tid];  // gradient wrt out[n, co, ho, wo]
    int idx  = argmax[tid];    // which (ci, kh, kw) was max?

    // decode n, co, ho, wo from tid
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

    // Recompute the input coordinate
    // consistent with forward: hi=(ho-padH)+kh, wi=(wo-padW)+kw_
    int hi = (ho - padH) + kh;
    int wi = (wo - padW) + kw_;

    // If hi, wi out-of-bounds, skip
    if (hi < 0 || hi >= Hin || wi < 0 || wi >= Win) {
        // means the max was an out-of-bounds location => if we used negInf, that can't happen
        // if we used zero, it could happen if everything was negative.
        // We'll just skip to keep it consistent with 'out-of-bounds => no gradient'
        return;
    }

    // Atomic adds
    atomicAdd(&grad_in[((n * Cin + ci) * Hin + hi) * Win + wi], go);
    atomicAdd(&grad_w[((co * Cin + ci) * Kh + kh) * Kw + kw_], go);
}

// Then the wrapper:
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
