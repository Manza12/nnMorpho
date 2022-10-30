#include "greyscale_operators.h"

// Erosion
template <typename scalar>
torch::Tensor erosion(torch::Tensor input,
                      torch::Tensor str_el,
                      torch::Tensor footprint,
                      int origin_x,
                      int origin_y,
                      char border_type,
                      scalar top,
                      scalar bottom) {

    // Compute output size
    const auto m = input.size(1);
    const auto n = input.size(0);
    const auto p = str_el.size(1);
    const auto q = str_el.size(0);

    // Initialization
    auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    torch::Tensor output_tensor = torch::zeros({n, m}, options);

    // Create accessors
    auto input_accessor = input.accessor<scalar, 2>();
    auto str_el_accessor = str_el.accessor<scalar, 2>();
    auto footprint_accessor = footprint.accessor<bool, 2>();
    auto output_accessor = output_tensor.accessor<scalar, 2>();

    scalar value;
    scalar difference;
    int idx_x, idx_y;
    // Computation
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < m; x++) {
            value = top;
            // Compute the value of output[y][x]
            for (int j = 0; j < q; j++) {
                for (int i = 0; i < p; i++) {
                    if (footprint_accessor[j][i]) {
                        idx_x = x + (i - origin_x);
                        idx_y = y + (j - origin_y);
                        if (0 <= idx_x && idx_x < m && 0 <= idx_y && idx_y < n) {
                            difference = input_accessor[idx_y][idx_x] - str_el_accessor[j][i];
                            if (value > difference) value = difference;
                        } else if (border_type == 'e') {
                            value = bottom;
                            goto end;
                        }
                    }
                }
            }
            end: output_accessor[y][x] = value;
        }
    }

    return output_tensor;
};

// Dilation
template <typename scalar>
torch::Tensor dilation(torch::Tensor input,
                       torch::Tensor str_el,
                       torch::Tensor footprint,
                       int origin_x,
                       int origin_y,
                       scalar top,
                       scalar bottom) {

    // Compute output size
    const auto m = input.size(1);
    const auto n = input.size(0);
    const auto p = str_el.size(1);
    const auto q = str_el.size(0);

    // Initialization
    auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
    torch::Tensor output_tensor = torch::zeros({n, m}, options);

    // Create accessors
    auto input_accessor = input.accessor<scalar, 2>();
    auto str_el_accessor = str_el.accessor<scalar, 2>();
    auto footprint_accessor = footprint.accessor<bool, 2>();
    auto output_accessor = output_tensor.accessor<scalar, 2>();

    scalar value;
    scalar sum;
    int idx_x, idx_y;
    // Computation
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < m; x++) {
            value = bottom;
            // Compute the value of output[y][x]
            for (int j = q-1; j >= 0; j--) {
                for (int i = p-1; i >= 0; i--) {
                    if (footprint_accessor[j][i]) {
                        idx_x = x - (i - origin_x);
                        idx_y = y - (j - origin_y);
                        if (0 <= idx_x && idx_x < m && 0 <= idx_y && idx_y < n) {
                            sum = input_accessor[idx_y][idx_x] + str_el_accessor[j][i];
                            if (value < sum) value = sum;
                        }
                    }
                }
            }
            output_accessor[y][x] = value;
        }
    }

    return output_tensor;
};
