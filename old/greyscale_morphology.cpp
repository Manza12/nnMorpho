#include <torch/extension.h>
#include <iostream>
#include <stdio.h>

#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_FLOAT(x)


// C++ interface
torch::Tensor greyscale_erosion(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor
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
	auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kFloat32);
	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options);	
	
	// Create accessors
	auto input_accessor = input_tensor.accessor<float, 2>();
	auto strel_accessor = strel_tensor.accessor<float, 2>();
	auto output_accessor = output_tensor.accessor<float, 2>();
	
	// Computation
	for (int x = 0; x < output_width; x++) {
		for (int y = 0; y < output_height; y++) {
			float value = INFINITY;
            float input_value;
            float structure_value;
            float difference;
			
			// Compute the value of output[y][x]			
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					input_value = input_accessor[x + i][y + j];
					structure_value = strel_accessor[i][j];
                    difference = input_value - structure_value;
                    if (value > difference)
                        value = difference;
				}
			}
			output_accessor[x][y] = value;
		}
	}
	
	return output_tensor;
}

torch::Tensor greyscale_dilation(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor
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
	auto options = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kFloat32);
	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options);	
	
	// Create accessors
	auto input_accessor = input_tensor.accessor<float, 2>();
	auto strel_accessor = strel_tensor.accessor<float, 2>();
	auto output_accessor = output_tensor.accessor<float, 2>();
	
	// Computation
	for (int x = 0; x < output_width; x++) {
		for (int y = 0; y < output_height; y++) {
            float value = -INFINITY;
            float input_value;
            float structure_value;
            float addition;
			
			// Compute the value of output[y][x]			
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
                    input_value = input_accessor[x + i][y + j];
                    structure_value = strel_accessor[strel_width - (i + 1)][strel_height - (j + 1)];
                    addition = input_value + structure_value;
                    if (value < addition)
                        value = addition;
				}
			}
			output_accessor[x][y] = value;
		}
	}
	
	return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("greyscale_erosion", &greyscale_erosion, "Greyscale Erosion (CPU)");
  m.def("greyscale_dilation", &greyscale_dilation, "Greyscale Dilation (CPU)");
}
