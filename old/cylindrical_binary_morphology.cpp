#include <torch/extension.h>
#include <iostream>
#include <stdio.h>

#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kBool, #x " must be a bool tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_FLOAT(x)


// C++ interface
torch::Tensor cylindric_binary_erosion(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		bool border_value,
		int origin_x,
		int origin_y
		) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	
	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	// Initialization
	auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kBool);
	torch::Tensor output_tensor = torch::zeros({input_width, input_height}, options);	
	
	// Create accessors
	auto input_accessor = input_tensor.accessor<bool,2>();
	auto strel_accessor = strel_tensor.accessor<bool,2>();
	auto output_accessor = output_tensor.accessor<bool,2>();
	
	// Computation
	int idx_x, idx_y;
	for (int x = 0; x < input_width; x++) {
		for (int y = 0; y < input_height; y++) {
			bool value = true;
			bool input_value;
			bool structure_value;
			
			// Compute the value of output[x][y]
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					idx_x = (x + i - origin_x) % input_width;
					if (idx_x < 0) {
						idx_x += input_width;
					}
					idx_y = y + j - origin_y;
					if (0 <= idx_y && idx_y < input_height) {
						input_value = input_accessor[idx_x][idx_y];
					} else {
						input_value = border_value;
					}
						
					structure_value = strel_accessor[i][j];
					if (structure_value && !input_value) {
						value = false;
						goto end;
					}
				}
			}
			end:
			output_accessor[x][y] = value;
		}
	}
	
	return output_tensor;
}

torch::Tensor cylindric_binary_dilation(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		bool border_value,
		int origin_x,
		int origin_y
		) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	
	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	// Initialization
	auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kBool);
	torch::Tensor output_tensor = torch::zeros({input_width, input_height}, options);	
	
	// Create accessors
	auto input_accessor = input_tensor.accessor<bool,2>();
	auto strel_accessor = strel_tensor.accessor<bool,2>();
	auto output_accessor = output_tensor.accessor<bool,2>();
	
	// Computation
	int idx_x, idx_y;
	for (int x = 0; x < input_width; x++) {
		for (int y = 0; y < input_height; y++) {
			bool value = false;
			bool input_value;
			bool structure_value;
			
			// Compute the value of output[x][y]
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					idx_x = (x - i + origin_x) % input_width;
					if(idx_x < 0) {
						idx_x += input_width;
					}
					idx_y = y - j + origin_y;
					if (0 <= idx_y && idx_y < input_height) {
						input_value = input_accessor[idx_x][idx_y];
					} else {
						input_value = border_value;
					}
						
					structure_value = strel_accessor[i][j];
					if (structure_value && input_value) {
						value = true;
						goto end;
					}
				}
			}
			end:
			output_accessor[x][y] = value;
		}
	}
	
	return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cylindric_binary_erosion", &cylindric_binary_erosion, "Cylindric Binary Erosion (CPU)");
  m.def("cylindric_binary_dilation", &cylindric_binary_dilation, "Cylindric Binary Dilation (CPU)");
}
