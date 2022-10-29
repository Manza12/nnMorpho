#include <torch/extension.h>
#include <iostream>
#include <stdio.h>


// C++ Erosion
template <typename scalar>
torch::Tensor erosion(
		torch::Tensor input,
		torch::Tensor str_el,
		torch::Tensor footprint,
		int origin_x,
		int origin_y,
		float border_value
		) {
	
	// Compute output size
	const auto m = input.size(1);
	const auto n = input.size(0);
	const auto p = str_el.size(1);
	const auto q = str_el.size(0);
	
	// Initialization
	auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
	torch::Tensor output_tensor = torch::zeros({n, m}, options);
	
	// Create accessors
	auto input_accessor = input.accessor<float, 2>();
	auto strel_accessor = str_el.accessor<float, 2>();
	auto footprint_accessor = footprint.accessor<float, 2>();
	auto output_accessor = output_tensor.accessor<float, 2>();

	float value;
    float input_value;
    float str_el_value;
    float difference;
    int idx_x, idx_y;
	// Computation
	for (int y = 0; y < n; y++) {
	    for (int x = 0; x < m; x++) {
            value = INFINITY;
			// Compute the value of output[y][x]			
			for (int j = 0; j < q; j++) {
				for (int i = 0; i < p; i++) {
				    if (footprint_accessor[j][i]) {
				        str_el_value = strel_accessor[j][i];
				        idx_x = x + i - origin_x;
				        idx_y = y + j - origin_y;
				        if (0 <= idx_x && idx_x < p && 0 <= idx_y && idx_y < q) {
				            input_value = input_accessor[idx_y][idx_x];
				        } else {
				            input_value = border_value;
				        }
				        difference = input_value - str_el_value;
				        if (value > difference) value = difference;
				    }
				}
			}
			output_accessor[y][x] = value;
		}
	}
	
	return output_tensor;
}

// C++ dilation
torch::Tensor dilation(
		torch::Tensor input,
		torch::Tensor str_el,
		torch::Tensor footprint,
		int origin_x,
		int origin_y,
		float border_value
		) {

	// Compute output size
	const auto m = input.size(1);
	const auto n = input.size(0);
	const auto p = str_el.size(1);
	const auto q = str_el.size(0);

	// Initialization
	auto options = torch::TensorOptions().device(input.device()).dtype(input.dtype());
	torch::Tensor output_tensor = torch::zeros({n, m}, options);

	// Create accessors
	auto input_accessor = input.accessor<float, 2>();
	auto strel_accessor = str_el.accessor<float, 2>();
	auto footprint_accessor = footprint.accessor<float, 2>();
	auto output_accessor = output_tensor.accessor<float, 2>();

	float value;
    float input_value;
    float str_el_value;
    float sum;
    int idx_x, idx_y;
	// Computation
	for (int y = 0; y < n; y++) {
	    for (int x = 0; x < m; x++) {
            value = -INFINITY;
			// Compute the value of output[y][x]
			for (int j = q-1; j >= 0; j--) {
				for (int i = p-1; i >= 0; i--) {
				    if (footprint_accessor[j][i]) {
				        str_el_value = strel_accessor[j][i];
				        idx_x = x - i + origin_x;
				        idx_y = y - j + origin_y;
				        if (0 <= idx_x && idx_x < p && 0 <= idx_y && idx_y < q) {
				            input_value = input_accessor[idx_y][idx_x];
				        } else {
				            input_value = border_value;
				        }
				        sum = input_value + str_el_value;
				        if (value < sum) value = sum;
				    }
				}
			}
			output_accessor[y][x] = value;
		}
	}
	
	return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("erosion", &erosion, "Erosion (CPU)");
  m.def("dilation", &dilation, "Dilation (CPU)");
}
