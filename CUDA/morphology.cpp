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

torch::Tensor dilation_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);
		
torch::Tensor erosion_batched_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);
		
torch::Tensor dilation_batched_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

torch::Tensor partial_erosion_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

torch::Tensor erosion_dependent_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

torch::Tensor dilation_dependent_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);
		
std::vector<torch::Tensor> erosion_forward_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

std::vector<torch::Tensor> erosion_backward_cuda(
		torch::Tensor grad_output,
		torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
		torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
		torch::Tensor block_shape);

std::vector<torch::Tensor> erosion_batched_forward_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

std::vector<torch::Tensor> erosion_batched_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape);
		
std::vector<torch::Tensor> dilation_forward_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

std::vector<torch::Tensor> dilation_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape);

std::vector<torch::Tensor> dilation_batched_forward_cuda(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape);

std::vector<torch::Tensor> dilation_batched_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape);

// C++ interface
torch::Tensor erosion(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor output_tensor = erosion_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

torch::Tensor dilation(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor output_tensor = dilation_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

torch::Tensor erosion_batched(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor output_tensor = erosion_batched_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

torch::Tensor dilation_batched(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor output_tensor = dilation_batched_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

torch::Tensor partial_erosion(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor output_tensor = partial_erosion_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

torch::Tensor erosion_dependent(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor output_tensor = erosion_dependent_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

torch::Tensor dilation_dependent(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	torch::Tensor output_tensor = dilation_dependent_cuda(input_tensor, strel_tensor, block_shape); 
	
	return output_tensor;
}

std::vector<torch::Tensor> erosion_forward(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	std::vector<torch::Tensor> outputs = erosion_forward_cuda(input_tensor, strel_tensor, block_shape); 
	
	return outputs;
}

std::vector<torch::Tensor> erosion_backward(
		torch::Tensor grad_output,
		torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
		torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(grad_output);
	CHECK_SHORT(strel_shape);
    CHECK_SHORT(origin_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
    std::vector<torch::Tensor> gradients = erosion_backward_cuda(
            grad_output, indexes_input, indexes_strel, strel_shape, origin_tensor, block_shape);
	
	return gradients;
}

std::vector<torch::Tensor> erosion_batched_forward(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	std::vector<torch::Tensor> outputs = erosion_batched_forward_cuda(input_tensor, strel_tensor, block_shape); 
	
	return outputs;
}

std::vector<torch::Tensor> erosion_batched_backward(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape) {

    // Checks
    CHECK_INPUT(grad_output);
    CHECK_SHORT(strel_shape);
    CHECK_SHORT(origin_tensor);
    CHECK_SHORT(block_shape);
	
	// Computation
    std::vector<torch::Tensor> gradients = erosion_batched_backward_cuda(
            grad_output, indexes_input, indexes_strel, strel_shape, origin_tensor, block_shape);
	
	return gradients;
}

std::vector<torch::Tensor> dilation_forward(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	std::vector<torch::Tensor> outputs = dilation_forward_cuda(input_tensor, strel_tensor, block_shape); 
	
	return outputs;
}

std::vector<torch::Tensor> dilation_backward(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape) {

    // Checks
    CHECK_INPUT(grad_output);
    CHECK_SHORT(strel_shape);
    CHECK_SHORT(origin_tensor);
    CHECK_SHORT(block_shape);
	
	// Computation
    std::vector<torch::Tensor> gradients = dilation_backward_cuda(
            grad_output, indexes_input, indexes_strel, strel_shape, origin_tensor, block_shape);
	
	return gradients;
}

std::vector<torch::Tensor> dilation_batched_forward(
		torch::Tensor input_tensor,
		torch::Tensor strel_tensor,
		torch::Tensor block_shape) {
	
	// Checks
	CHECK_INPUT(input_tensor);
	CHECK_INPUT(strel_tensor);
	CHECK_SHORT(block_shape);
	
	// Computation
	std::vector<torch::Tensor> outputs = dilation_batched_forward_cuda(input_tensor, strel_tensor, block_shape); 
	
	return outputs;
}

std::vector<torch::Tensor> dilation_batched_backward(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape) {

    // Checks
    CHECK_INPUT(grad_output);
    CHECK_SHORT(strel_shape);
    CHECK_SHORT(origin_tensor);
    CHECK_SHORT(block_shape);

    // Computation
    std::vector<torch::Tensor> gradients = dilation_batched_backward_cuda(
            grad_output, indexes_input, indexes_strel, strel_shape, origin_tensor, block_shape);

    return gradients;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("erosion", &erosion, "Erosion (CUDA)");
  m.def("dilation", &dilation, "Dilation (CUDA)");
  
  m.def("erosion_batched", &erosion_batched, "Erosion batched (CUDA)");
  m.def("dilation_batched", &dilation_batched, "Dilation batched (CUDA)");
  
  m.def("partial_erosion", &partial_erosion, "Partial erosion (CUDA)");
  
  m.def("erosion_dependent", &erosion_dependent, "Erosion dependent (CUDA)");
  m.def("dilation_dependent", &dilation_dependent, "Dilation dependent (CUDA)");
  
  m.def("erosion_forward", &erosion_forward, "Erosion forward (CUDA)");
  m.def("erosion_backward", &erosion_backward, "Erosion backward (CUDA)");
  
  m.def("erosion_batched_forward", &erosion_batched_forward, "Erosion batched forward (CUDA)");
  m.def("erosion_batched_backward", &erosion_batched_backward, "Erosion batched backward (CUDA)");
  
  m.def("dilation_forward", &dilation_forward, "Dilation forward (CUDA)");
  m.def("dilation_backward", &dilation_backward, "Dilation backward (CUDA)");

  m.def("dilation_batched_forward", &dilation_batched_forward, "Dilation batched forward (CUDA)");
  m.def("dilation_batched_backward", &dilation_batched_backward, "Dilation batched backward (CUDA)");
}

