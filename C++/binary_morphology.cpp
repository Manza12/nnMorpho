#include <torch/extension.h>
#include <iostream>
#include <stdio.h>

#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == torch::kBool, #x " must be a bool tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_BOOL(x)


// C++ interface
torch::Tensor binary_erosion(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
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

	const auto output_width = input_width - strel_width + 1;
	const auto output_height = input_height - strel_height + 1;
	
	// Initialization
	auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kBool);
	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options);	
	
	// Create accessors
	auto input_accessor = input_tensor.accessor<bool,2>();
	auto strel_accessor = strel_tensor.accessor<bool,2>();
	auto output_accessor = output_tensor.accessor<bool,2>();
	
	// Computation
	for (int x = 0; x < output_width; x++) {
		for (int y = 0; y < output_height; y++) {
			bool value = true;
			bool input_value;
			bool structure_value;
			
			// Compute the value of output[y][x]			
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					input_value = input_accessor[x + i][y + j];
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

torch::Tensor binary_dilation(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
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

	const auto output_width = input_width - strel_width + 1;
	const auto output_height = input_height - strel_height + 1;
	
	// Initialization
	auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kBool);
	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options);	
	
	// Create accessors
	auto input_accessor = input_tensor.accessor<bool,2>();
	auto strel_accessor = strel_tensor.accessor<bool,2>();
	auto output_accessor = output_tensor.accessor<bool,2>();
	
	// Computation
	for (int x = 0; x < output_width; x++) {
		for (int y = 0; y < output_height; y++) {
			bool value = false;
			bool input_value;
			bool structure_value;
			
			// Compute the value of output[y][x]			
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					input_value = input_accessor[x - i + strel_width - 1][y - j + strel_height - 1];
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
  m.def("binary_erosion", &binary_erosion, "Binary Erosion (CPU)");
  m.def("binary_dilation", &binary_dilation, "Binary Dilation (CPU)");
}
