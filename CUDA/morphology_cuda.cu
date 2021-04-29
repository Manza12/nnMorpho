#include <torch/extension.h>

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <float.h>
#include <assert.h>

#include <iostream>
#include <stdio.h>

// Macros
#define INF FLT_MAX


/* CUDA kernels */
__global__ void erosion_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor) {
	
	/* Sizes */
	// Input
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	
	// Strel
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	// Output
	const auto output_width = output_tensor.size(0);
	const auto output_height = output_tensor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value = INF;
	float candidate;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
		for (int j = 0; j < strel_height; j++) {
			for (int i = 0; i < strel_width; i++) {
				candidate = input_tensor[x + i][y + j] - strel_tensor[i][j];
				if (candidate < value) {
					value = candidate;
				}
			}
		}
		output_tensor[x][y] = value;
	}
}

__global__ void dilation_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor) {
	
	/* Sizes */
	// Input
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	
	// Strel
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	// Output
	const auto output_width = output_tensor.size(0);
	const auto output_height = output_tensor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value = -INF;
	float candidate;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
		for (int j = 0; j < strel_height; j++) {
			for (int i = 0; i < strel_width; i++) {
				candidate = input_tensor[x + i][y + j] + strel_tensor[strel_width - (i + 1)][strel_height - (j + 1)];
				if (candidate > value) {
					value = candidate;
				}
			}
		}
		output_tensor[x][y] = value;
	}
}

__global__ void erosion_batched_cuda_kernel(
		const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,3> output_tensor) {
	
	/* Sizes */
	// Input
	const auto batch_size = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto input_width = input_tensor.size(2);
	
	// Strel
	const auto strel_height = strel_tensor.size(0);
	const auto strel_width = strel_tensor.size(1);
	
	// Output
	const auto output_height = output_tensor.size(1);
	const auto output_width = output_tensor.size(2);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value;
	float candidate;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
		for (int k = 0; k < batch_size; k++) {
			value = INF;
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					candidate = input_tensor[k][y + j][x + i] - strel_tensor[j][i];
					if (candidate < value) {
						value = candidate;
					}
				}
			}
			output_tensor[k][y][x] = value;
		}
	}
}

__global__ void dilation_batched_cuda_kernel(
		const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,3> output_tensor) {
	
	/* Sizes */
	// Input
	const auto batch_size = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto input_width = input_tensor.size(2);
	
	// Strel
	const auto strel_height = strel_tensor.size(0);
	const auto strel_width = strel_tensor.size(1);
	
	// Output
	const auto output_height = output_tensor.size(1);
	const auto output_width = output_tensor.size(2);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value;
	float candidate;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
		for (int k = 0; k < batch_size; k++) {
			value = -INF;
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					candidate = input_tensor[k][y + j][x + i] + strel_tensor[strel_height - (j + 1)][strel_width - (i + 1)];
					if (candidate > value) {
						value = candidate;
					}
				}
			}
			output_tensor[k][y][x] = value;
		}
	}
}

__global__ void partial_erosion_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor) {
	
	/* Sizes */
	// Input
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	
	// Strel
	const auto strel_height = strel_tensor.size(1);
	
	// Output
	const auto output_width = output_tensor.size(0);
	const auto output_height = output_tensor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value = INF;
	float candidate;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {		
		for (int j = 0; j < strel_height; j++) {
			candidate = input_tensor[x][y + j] - strel_tensor[x][j];
			if (candidate < value) {
				value = candidate;
			}
		}
		output_tensor[x][y] = value;
	}
}

__global__ void erosion_forward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor,
		torch::PackedTensorAccessor32<short,3> indexes) {
	
	/* Sizes */
	// Input
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	
	// Strel
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	// Output
	const auto output_width = output_tensor.size(0);
	const auto output_height = output_tensor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value = INF;
	float candidate;
	int index_i;
	int index_j;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
		for (int i = 0; i < strel_width; i++) {
			for (int j = 0; j < strel_height; j++) {
				candidate = input_tensor[x + i][y + j] - strel_tensor[i][j];
				if (candidate < value) {
					value = candidate;
					index_i = i;
					index_j = j;
				}
			}
		}
		output_tensor[x][y] = value;
		indexes[x][y][0] = index_i;
		indexes[x][y][1] = index_j;
	}
}

__global__ void erosion_backward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_output_accessor,
		const torch::PackedTensorAccessor32<short,3,torch::RestrictPtrTraits> indexes_accessor,
		torch::PackedTensorAccessor32<float,2> grad_input_accessor) {
	
	/* Sizes */
	// Grad Output
	const auto grad_output_width = grad_output_accessor.size(0);
	const auto grad_output_height = grad_output_accessor.size(1);
	
	// Indexes
	const auto indexes_width = indexes_accessor.size(0);
	const auto indexes_height = indexes_accessor.size(1);
	
	// Grad Input
	const auto grad_input_width = grad_input_accessor.size(0);
	const auto grad_input_height = grad_input_accessor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Add the value to the grad_input_accessor
	if (x < grad_output_width && y < grad_output_height) {
		short index_i = indexes_accessor[x][y][0];
		short index_j = indexes_accessor[x][y][1];
		atomicAdd(&grad_input_accessor[index_i][index_j], -grad_output_accessor[x][y]);
	}
}

__global__ void dilation_forward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor,
		torch::PackedTensorAccessor32<short,3> indexes) {
	
	/* Sizes */
	// Input
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	
	// Strel
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	// Output
	const auto output_width = output_tensor.size(0);
	const auto output_height = output_tensor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value = -INF;
	float candidate;
	int index_i;
	int index_j;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
		for (int j = 0; j < strel_height; j++) {
			for (int i = 0; i < strel_width; i++) {
				candidate = input_tensor[x + i][y + j] + strel_tensor[strel_width - (i + 1)][strel_height - (j + 1)];
				if (candidate > value) {
					value = candidate;
					index_i = strel_width - (i + 1);
					index_j = strel_height - (j + 1);
				}
			}
		}
		output_tensor[x][y] = value;
		indexes[x][y][0] = index_i;
		indexes[x][y][1] = index_j;
	}
}

__global__ void dilation_backward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_output_accessor,
		const torch::PackedTensorAccessor32<short,3,torch::RestrictPtrTraits> indexes_accessor,
		torch::PackedTensorAccessor32<float,2> grad_input_accessor) {
	
	/* Sizes */
	// Grad Output
	const auto grad_output_width = grad_output_accessor.size(0);
	const auto grad_output_height = grad_output_accessor.size(1);
	
	// Indexes
	const auto indexes_width = indexes_accessor.size(0);
	const auto indexes_height = indexes_accessor.size(1);
	
	// Grad Input
	const auto grad_input_width = grad_input_accessor.size(0);
	const auto grad_input_height = grad_input_accessor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Add the value to the grad_input_accessor
	if (x < grad_output_width && y < grad_output_height) {
		short index_i = indexes_accessor[x][y][0];
		short index_j = indexes_accessor[x][y][1];
		atomicAdd(&grad_input_accessor[index_i][index_j], grad_output_accessor[x][y]);
	}
}

/* CUDA */
torch::Tensor erosion_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	const auto output_width = input_width - strel_width + 1;
	const auto output_height = input_height - strel_height + 1;
  	
  	// Initialize output tensor
  	auto options = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((output_width - 1) / block_width) + 1;
	const int grid_height = ((output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,2>();

	// Launch of the kernel
	erosion_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor);
	
  	return output_tensor;
}

torch::Tensor dilation_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	const auto output_width = input_width - strel_width + 1;
	const auto output_height = input_height - strel_height + 1;
  	
  	// Initialize output tensor
  	auto options = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((output_width - 1) / block_width) + 1;
	const int grid_height = ((output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,2>();

	// Launch of the kernel
	dilation_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor);
	
  	return output_tensor;
}

torch::Tensor erosion_batched_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto batch_size = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto input_width = input_tensor.size(2);
	const auto strel_height = strel_tensor.size(0);
	const auto strel_width = strel_tensor.size(1);
	
	const auto output_height = input_height - strel_height + 1;
	const auto output_width = input_width - strel_width + 1;
  	
  	// Initialize output tensor
  	auto options = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({batch_size, output_height, output_width}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_height = block_ptr[0];
  	const short block_width = block_ptr[1];
  	
	const int grid_height = ((output_height - 1) / block_height) + 1;
	const int grid_width = ((output_width - 1) / block_width) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,3>();

	// Launch of the kernel
	erosion_batched_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor);
	
  	return output_tensor;
}

torch::Tensor dilation_batched_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto batch_size = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto input_width = input_tensor.size(2);
	const auto strel_height = strel_tensor.size(0);
	const auto strel_width = strel_tensor.size(1);
	
	const auto output_height = input_height - strel_height + 1;
	const auto output_width = input_width - strel_width + 1;
  	
  	// Initialize output tensor
  	auto options = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({batch_size, output_height, output_width}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_height = block_ptr[0];
  	const short block_width = block_ptr[1];
  	
	const int grid_height = ((output_height - 1) / block_height) + 1;
	const int grid_width = ((output_width - 1) / block_width) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,3>();

	// Launch of the kernel
	dilation_batched_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor);
	
  	return output_tensor;
}

torch::Tensor partial_erosion_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_height = strel_tensor.size(1);

	const auto output_width = input_width;
	const auto output_height = input_height - strel_height + 1;
	
  	// Initialize output tensor
  	auto options = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((output_width - 1) / block_width) + 1;
	const int grid_height = ((output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,2>();

	// Launch of the kernel
	partial_erosion_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor);
	
  	return output_tensor;
}

std::vector<torch::Tensor> erosion_forward_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	const auto output_width = input_width - strel_width + 1;
	const auto output_height = input_height - strel_height + 1;
  	
  	// Initialize output tensor
  	auto options_output = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options_output);
  	
  	// Initialize indexes
  	auto options_indexes = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt16);
  	torch::Tensor indexes = torch::zeros({output_width, output_height, 2}, options_indexes);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((output_width - 1) / block_width) + 1;
	const int grid_height = ((output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,2>();
	auto indexes_accessor = indexes.packed_accessor32<short,3>();

	// Launch of the kernel
	erosion_forward_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor, indexes_accessor);
	
  	return {output_tensor, indexes};
}


torch::Tensor erosion_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor indexes,
    torch::Tensor strel_shape,
    torch::Tensor block_shape) {

	// Compute output size
	const auto grad_output_width = grad_output.size(0);
	const auto grad_output_height = grad_output.size(1);
	
	// Compute Grad Input size
	short* strel_ptr = strel_shape.data_ptr<short>();
  	const short strel_width = strel_ptr[0];
  	const short strel_height = strel_ptr[1];
	
  	// Initialize output tensor
  	auto options = torch::TensorOptions().device(grad_output.device());
  	torch::Tensor grad_input = torch::zeros({strel_width, strel_height}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((grad_output_width - 1) / block_width) + 1;
	const int grid_height = ((grad_output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto grad_output_accessor = grad_output.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto indexes_accessor = indexes.packed_accessor32<short,3,torch::RestrictPtrTraits>();
	auto grad_input_accessor = grad_input.packed_accessor32<float,2>();

	// Launch of the kernel
	erosion_backward_cuda_kernel<<<grid_size, block_size>>>(grad_output_accessor, indexes_accessor, grad_input_accessor);
	
  	return grad_input;
}

std::vector<torch::Tensor> dilation_forward_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(0);
	const auto strel_height = strel_tensor.size(1);
	
	const auto output_width = input_width - strel_width + 1;
	const auto output_height = input_height - strel_height + 1;
  	
  	// Initialize output tensor
  	auto options_output = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options_output);
  	
  	// Initialize indexes
  	auto options_indexes = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt16);
  	torch::Tensor indexes = torch::zeros({output_width, output_height, 2}, options_indexes);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((output_width - 1) / block_width) + 1;
	const int grid_height = ((output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,2>();
	auto indexes_accessor = indexes.packed_accessor32<short,3>();

	// Launch of the kernel
	dilation_forward_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor, indexes_accessor);
	
  	return {output_tensor, indexes};
}


torch::Tensor dilation_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor indexes,
    torch::Tensor strel_shape,
    torch::Tensor block_shape) {

	// Compute output size
	const auto grad_output_width = grad_output.size(0);
	const auto grad_output_height = grad_output.size(1);
	
	// Compute Grad Input size
	short* strel_ptr = strel_shape.data_ptr<short>();
  	const short strel_width = strel_ptr[0];
  	const short strel_height = strel_ptr[1];
	
  	// Initialize output tensor
  	auto options = torch::TensorOptions().device(grad_output.device());
  	torch::Tensor grad_input = torch::zeros({strel_width, strel_height}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((grad_output_width - 1) / block_width) + 1;
	const int grid_height = ((grad_output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto grad_output_accessor = grad_output.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto indexes_accessor = indexes.packed_accessor32<short,3,torch::RestrictPtrTraits>();
	auto grad_input_accessor = grad_input.packed_accessor32<float,2>();

	// Launch of the kernel
	dilation_backward_cuda_kernel<<<grid_size, block_size>>>(grad_output_accessor, indexes_accessor, grad_input_accessor);
	
  	return grad_input;
}
