#include <torch/extension.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_SHORT(x) TORCH_CHECK(x.scalar_type() == torch::kInt16, #x " must be a int16 tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_FLOAT(x)

// CUDA declarations
torch::Tensor erosion_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

std::vector<torch::Tensor> erosion_forward_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);
		
torch::Tensor erosion_backward_cuda(
		torch::Tensor grad_output,
		torch::Tensor indexes,
		torch::Tensor strel_shape,
		torch::Tensor block_shape);

// C++ interface
torch::Tensor erosion(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	
	// Computation
	torch::Tensor output_tensor = erosion_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

std::vector<torch::Tensor> erosion_forward(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	
	// Computation
	std::vector<torch::Tensor> outputs = erosion_forward_cuda(input_tensor, strel_tensor, block_shape); 
	
	return outputs;
}

torch::Tensor erosion_backward(
		torch::Tensor grad_output,
		torch::Tensor indexes,
		torch::Tensor strel_shape,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(grad_output);
	CHECK_SHORT(strel_shape);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor grad_input = erosion_backward_cuda(grad_output, indexes, strel_shape, block_shape); 
	
	return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("erosion", &erosion, "Erosion (CUDA)");
  m.def("erosion_forward", &erosion_forward, "Erosion forward (CUDA)");
  m.def("erosion_backward", &erosion_backward, "Erosion backward (CUDA)");
}
