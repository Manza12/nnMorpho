#include <cuda.h>
#include <cuda_runtime.h>
#include "binary_operators.cpp"

/* Kernels */
// Erosion
template <typename scalar>
__global__ void erosion_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar, 2, torch::RestrictPtrTraits> input_accessor,
        const torch::PackedTensorAccessor32<scalar, 2, torch::RestrictPtrTraits> str_el_accessor,
        torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> output_accessor,
        const int origin_x,
        const int origin_y,
        const char border_type) {

    // Sizes
    const auto m = input_accessor.size(1);
    const auto n = input_accessor.size(0);
    const auto p = str_el_accessor.size(1);
    const auto q = str_el_accessor.size(0);

    // Compute thread index corresponding in output tensor
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare variables
    bool value;
    int idx_x, idx_y;

    // Compute the value of output[y][x]
    if (x < m && y < n) {
        value = true;
        for (int j = 0; j < q; j++) {
            for (int i = 0; i < p; i++) {
                idx_x = x + (i - origin_x);
                idx_y = y + (j - origin_y);
                if (0 <= idx_x && idx_x < m && 0 <= idx_y && idx_y < n) {
                    if (str_el_accessor[j][i] > input_accessor[idx_y][idx_x]) {
                        value = false;
                        goto end;
                    }
                } else if (border_type == 'e') {
                    if (str_el_accessor[j][i]) {
                        value = false;
                        goto end;
                    }
                }
            }
        }
        end: output_accessor[y][x] = value;
    }
}

// Dilation
template <typename scalar>
__global__ void dilation_cuda_kernel(
        const torch::PackedTensorAccessor32<bool, 2, torch::RestrictPtrTraits> input_accessor,
        const torch::PackedTensorAccessor32<scalar, 2, torch::RestrictPtrTraits> str_el_accessor,
        torch::PackedTensorAccessor32<scalar, 2, torch::RestrictPtrTraits> output_accessor,
        const int origin_x,
        const int origin_y,
        const scalar bottom) {

    // Sizes
    const auto m = input_accessor.size(1);
    const auto n = input_accessor.size(0);
    const auto p = str_el_accessor.size(1);
    const auto q = str_el_accessor.size(0);

    // Compute thread index corresponding in output tensor
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare variables
    scalar value = bottom;
    scalar current;
    int idx_x, idx_y;

    // Compute the value of output[y][x]
    if (x < m && y < n) {
        for (int j = q-1; j >= 0; j--) {
            for (int i = p-1; i >= 0; i--) {
                idx_x = x - (i - origin_x);
                idx_y = y - (j - origin_y);
                if (0 <= idx_x && idx_x < m && 0 <= idx_y && idx_y < n) {
                    if (input_accessor[idx_y][idx_x]) {
                        current = str_el_accessor[j][i];
                        if (value < current) value = current;
                    }
                }
            }
        }
        output_accessor[y][x] = value;
    }
}

/* Implementations */
// Erosion
template <typename scalar>
torch::Tensor erosion(torch::Tensor input,
                      torch::Tensor str_el,
                      int origin_y,
                      int origin_x,
                      char border_type,
                      const int block_size_y,
                      const int block_size_x) {

    // Compute output size
    const auto m = input.size(1);
    const auto n = input.size(0);
    const auto p = str_el.size(1);
    const auto q = str_el.size(0);

    // Initialization
    auto options = torch::TensorOptions().device(input.device()).dtype(torch::ScalarType::Bool);
    torch::Tensor output_tensor = torch::zeros({n, m}, options);

    // Switch between CPU and GPU
    if (input.is_cuda()) {
        /* GPU */
        // Create accessors
        auto input_accessor = input.packed_accessor32<scalar, 2, torch::RestrictPtrTraits>();
        auto str_el_accessor = str_el.packed_accessor32<scalar, 2, torch::RestrictPtrTraits>();
        auto output_accessor = output_tensor.packed_accessor32<bool, 2, torch::RestrictPtrTraits>();

        // Block & Grid parameters
        const int grid_x = ((m - 1) / block_size_x) + 1;
        const int grid_y = ((n - 1) / block_size_y) + 1;

        const dim3 block_size(block_size_x, block_size_y, 1);
        const dim3 grid_size(grid_x, grid_y, 1);

        // Launch of the kernel
        erosion_cuda_kernel<<<grid_size, block_size>>>(
                input_accessor,
                str_el_accessor,
                output_accessor,
                origin_x, origin_y, border_type);
    } else {
        /* CPU */
        // Create accessors
        auto input_accessor = input.accessor<scalar, 2>();
        auto str_el_accessor = str_el.accessor<scalar, 2>();
        auto output_accessor = output_tensor.accessor<bool, 2>();

        // Declare variables
        bool value;
        int idx_x, idx_y;

        // Computation
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < m; x++) {
                value = true;
                // Compute the value of output[y][x]
                for (int j = 0; j < q; j++) {
                    for (int i = 0; i < p; i++) {
                        idx_x = x + (i - origin_x);
                        idx_y = y + (j - origin_y);
                        if (0 <= idx_x && idx_x < m && 0 <= idx_y && idx_y < n) {
                            if (str_el_accessor[j][i] > input_accessor[idx_y][idx_x]) {
                                value = false;
                                goto end;
                            }
                        } else if (border_type == 'e') {
                            if (str_el_accessor[j][i]) {
                                value = false;
                                goto end;
                            }
                        }
                    }
                }
                end: output_accessor[y][x] = value;
            }
        }
    }

    return output_tensor;
}

// Dilation
template <typename scalar>
torch::Tensor dilation(torch::Tensor input,
                       torch::Tensor str_el,
                       int origin_y,
                       int origin_x,
                       scalar bottom,
                       int block_size_y,
                       int block_size_x) {

    // Compute output size
    const auto m = input.size(1);
    const auto n = input.size(0);
    const auto p = str_el.size(1);
    const auto q = str_el.size(0);

    // Initialization
    auto options = torch::TensorOptions().device(input.device()).dtype(str_el.dtype());
    torch::Tensor output_tensor = torch::zeros({n, m}, options);

    // Switch between CPU and GPU
    if (input.is_cuda()) {
        /* GPU */
        // Create accessors
        auto input_accessor = input.packed_accessor32<bool, 2, torch::RestrictPtrTraits>();
        auto str_el_accessor = str_el.packed_accessor32<scalar, 2, torch::RestrictPtrTraits>();
        auto output_accessor = output_tensor.packed_accessor32<scalar, 2, torch::RestrictPtrTraits>();

        // Block & Grid parameters
        const int grid_x = ((m - 1) / block_size_x) + 1;
        const int grid_y = ((n - 1) / block_size_y) + 1;

        const dim3 block_size(block_size_x, block_size_y, 1);
        const dim3 grid_size(grid_x, grid_y, 1);

        // Launch of the kernel
        dilation_cuda_kernel<<<grid_size, block_size>>>(
                input_accessor,
                str_el_accessor,
                output_accessor,
                origin_x, origin_y, bottom);
    } else {
        /* CPU */
        // Create accessors
        auto input_accessor = input.accessor<bool, 2>();
        auto str_el_accessor = str_el.accessor<scalar, 2>();
        auto output_accessor = output_tensor.accessor<scalar, 2>();

        // Declare variables
        scalar value;
        scalar current;
        int idx_x, idx_y;

        // Computation
        for (int y = 0; y < n; y++) {
            for (int x = 0; x < m; x++) {
                value = bottom;
                // Compute the value of output[y][x]
                for (int j = q-1; j >= 0; j--) {
                    for (int i = p-1; i >= 0; i--) {
                        idx_x = x - (i - origin_x);
                        idx_y = y - (j - origin_y);
                        if (0 <= idx_x && idx_x < m && 0 <= idx_y && idx_y < n) {
                            if (input_accessor[idx_y][idx_x]) {
                                current = str_el_accessor[j][i];
                                if (value < current) value = current;
                            }
                        }
                    }
                }
                output_accessor[y][x] = value;
            }
        }
    }

    return output_tensor;
}
