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

__global__ void erosion_dependent_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor) {
	
	/* Sizes */
	// Input
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	
	// Strel
	const auto strel_width = strel_tensor.size(1);
	const auto strel_height = strel_tensor.size(2);
	
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
				candidate = input_tensor[x + i][y + j] - strel_tensor[x][i][j];
				if (candidate < value) {
					value = candidate;
				}
			}
		}
		output_tensor[x][y] = value;
	}
}

__global__ void dilation_dependent_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor) {
	
	/* Sizes */
	// Input
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	
	// Strel
	const auto strel_width = strel_tensor.size(1);
	const auto strel_height = strel_tensor.size(2);
	
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
				candidate = input_tensor[x + i][y + j] + strel_tensor[x][strel_width - (i + 1)][strel_height - (j + 1)];
				if (candidate > value) {
					value = candidate;
				}
			}
		}
		output_tensor[x][y] = value;
	}
}

__global__ void erosion_forward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,2> output_tensor,
		torch::PackedTensorAccessor32<short,3> indexes_input,
        torch::PackedTensorAccessor32<short,3> indexes_strel) {
	
	/* Sizes */
	// Strel
	const auto strel_height = strel_tensor.size(0);
    const auto strel_width = strel_tensor.size(1);

	// Output
	const auto output_height = output_tensor.size(0);
    const auto output_width = output_tensor.size(1);

	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value;
	float candidate;
	int index_i;
	int index_j;
    int index_x;
    int index_y;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
        value = INF;
		for (int j = 0; j < strel_height; j++) {
            for (int i = 0; i < strel_width; i++) {
				candidate = input_tensor[y + j][x + i] - strel_tensor[j][i];
				if (candidate < value) {
					value = candidate;
					index_i = i;
					index_j = j;
                    index_x = x + i;
                    index_y = y + j;
				}
			}
		}
		output_tensor[y][x] = value;
        indexes_strel[y][x][0] = index_i;
        indexes_strel[y][x][1] = index_j;
        indexes_input[y][x][0] = index_x;
        indexes_input[y][x][1] = index_y;
	}
}

__global__ void erosion_backward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_output_accessor,
		const torch::PackedTensorAccessor32<short,3,torch::RestrictPtrTraits> indexes_input_accessor,
        const torch::PackedTensorAccessor32<short,3,torch::RestrictPtrTraits> indexes_strel_accessor,
		torch::PackedTensorAccessor32<float,2> grad_input_accessor,
        torch::PackedTensorAccessor32<float,2> grad_strel_accessor,
        const short origin_height,
        const short origin_width) {
	
	/* Sizes */
	// Grad Output
	const auto grad_output_height = grad_output_accessor.size(0);
    const auto grad_output_width = grad_output_accessor.size(1);

	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Add the value to the gradients
	if (x < grad_output_width && y < grad_output_height) {
	    // grad_input
		short index_x = indexes_input_accessor[y][x][0];
		short index_y = indexes_input_accessor[y][x][1];
		if (origin_width <= index_x && index_x < grad_output_width + origin_width &&
                origin_height <= index_y && index_y < grad_output_height + origin_height) {
            atomicAdd(&grad_input_accessor[index_y - origin_height][index_x - origin_width], grad_output_accessor[y][x]);
		}

        // grad_strel
        short index_i = indexes_strel_accessor[y][x][0];
        short index_j = indexes_strel_accessor[y][x][1];
        atomicAdd(&grad_strel_accessor[index_j][index_i], -grad_output_accessor[y][x]);
	}
}

__global__ void erosion_batched_forward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
		torch::PackedTensorAccessor32<float,3> output_tensor,
		torch::PackedTensorAccessor32<short,4> indexes_input,
        torch::PackedTensorAccessor32<short,4> indexes_strel) {
	
	/* Sizes */
	// Strel
	const auto strel_height = strel_tensor.size(0);
	const auto strel_width = strel_tensor.size(1);
	
	// Output
	const auto batch_size = output_tensor.size(0);
	const auto output_height = output_tensor.size(1);
	const auto output_width = output_tensor.size(2);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value;
	float candidate;
	int index_i;
	int index_j;
    int index_x;
    int index_y;
	
	// Compute the value of output[k][y][x]
	if (x < output_width && y < output_height) {
		for (int k = 0; k < batch_size; k++) {
			value = INF;
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					candidate = input_tensor[k][y + j][x + i] - strel_tensor[j][i];
					if (candidate < value) {
						value = candidate;
						index_i = i;
						index_j = j;
                        index_x = x + i;
                        index_y = y + j;
					}
				}
			}
			output_tensor[k][y][x] = value;
			indexes_strel[k][y][x][0] = index_i;
			indexes_strel[k][y][x][1] = index_j;
            indexes_input[k][y][x][0] = index_x;
            indexes_input[k][y][x][1] = index_y;
		}
	}
}

__global__ void erosion_batched_backward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output_accessor,
		const torch::PackedTensorAccessor32<short,4,torch::RestrictPtrTraits> indexes_input_accessor,
        const torch::PackedTensorAccessor32<short,4,torch::RestrictPtrTraits> indexes_strel_accessor,
		torch::PackedTensorAccessor32<float,3> grad_input_accessor,
        torch::PackedTensorAccessor32<float,2> grad_strel_accessor,
        const short origin_height,
        const short origin_width) {

    /* Sizes */
	// Size of grad output
	const auto batch_size = grad_output_accessor.size(0);
	const auto grad_output_height = grad_output_accessor.size(1);
	const auto grad_output_width = grad_output_accessor.size(2);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Add the value to the gradients
	if (x < grad_output_width && y < grad_output_height) {
        // grad_input
        short index_x;
        short index_y;
        for (int k = 0; k < batch_size; k++) {
            index_x = indexes_input_accessor[k][y][x][0];
            index_y = indexes_input_accessor[k][y][x][1];
            if (origin_width <= index_x && index_x < grad_output_width + origin_width &&
                origin_height <= index_y && index_y < grad_output_height + origin_height) {
                atomicAdd(&grad_input_accessor[k][index_y - origin_height][index_x - origin_width],
                          grad_output_accessor[k][y][x]);
            }
        }

        // grad_strel
		short index_i;
		short index_j;
		for (int k = 0; k < batch_size; k++) {
			index_i = indexes_strel_accessor[k][y][x][0];
			index_j = indexes_strel_accessor[k][y][x][1];
			atomicAdd(&grad_strel_accessor[index_j][index_i], -grad_output_accessor[k][y][x]);
		}
	}
}

__global__ void dilation_forward_cuda_kernel(
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> input_tensor,
		const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
        torch::PackedTensorAccessor32<float,2> output_tensor,
        torch::PackedTensorAccessor32<short,3> indexes_input,
        torch::PackedTensorAccessor32<short,3> indexes_strel) {
	
	/* Sizes */
	// Strel
	const auto strel_height = strel_tensor.size(0);
    const auto strel_width = strel_tensor.size(1);

	// Output
	const auto output_height = output_tensor.size(0);
    const auto output_width = output_tensor.size(1);

	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value;
	float candidate;
	int index_i;
	int index_j;
    int index_x;
    int index_y;
	
	// Compute the value of output[y][x]
	if (x < output_width && y < output_height) {
        value = -INF;
		for (int j = 0; j < strel_height; j++) {
			for (int i = 0; i < strel_width; i++) {
				candidate = input_tensor[y + j][x + i] + strel_tensor[strel_height - (j + 1)][strel_width - (i + 1)];
				if (candidate > value) {
					value = candidate;
					index_i = strel_width - (i + 1);
					index_j = strel_height - (j + 1);
                    index_x = x + i;
                    index_y = y + j;
				}
			}
		}
        output_tensor[y][x] = value;
        indexes_strel[y][x][0] = index_i;
        indexes_strel[y][x][1] = index_j;
        indexes_input[y][x][0] = index_x;
        indexes_input[y][x][1] = index_y;
	}
}

__global__ void dilation_backward_cuda_kernel(
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_output_accessor,
        const torch::PackedTensorAccessor32<short,3,torch::RestrictPtrTraits> indexes_input_accessor,
        const torch::PackedTensorAccessor32<short,3,torch::RestrictPtrTraits> indexes_strel_accessor,
        torch::PackedTensorAccessor32<float,2> grad_input_accessor,
        torch::PackedTensorAccessor32<float,2> grad_strel_accessor,
        const short origin_height,
        const short origin_width) {
	
	/* Sizes */
	// Grad Output
	const auto grad_output_height = grad_output_accessor.size(0);
    const auto grad_output_width = grad_output_accessor.size(1);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Add the value to the gradients
	if (x < grad_output_width && y < grad_output_height) {
        // grad_input
        short index_x = indexes_input_accessor[y][x][0];
        short index_y = indexes_input_accessor[y][x][1];
        if (origin_width <= index_x && index_x < grad_output_width + origin_width &&
            origin_height <= index_y && index_y < grad_output_height + origin_height) {
            atomicAdd(&grad_input_accessor[index_y - origin_height][index_x - origin_width], grad_output_accessor[y][x]);
        }

        // grad_strel
        short index_i = indexes_strel_accessor[y][x][0];
        short index_j = indexes_strel_accessor[y][x][1];
        atomicAdd(&grad_strel_accessor[index_j][index_i], grad_output_accessor[y][x]);
	}
}

__global__ void dilation_batched_forward_cuda_kernel(
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> input_tensor,
        const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> strel_tensor,
        torch::PackedTensorAccessor32<float,3> output_tensor,
        torch::PackedTensorAccessor32<short,4> indexes_input,
        torch::PackedTensorAccessor32<short,4> indexes_strel) {
	
	/* Sizes */
	// Strel
	const auto strel_height = strel_tensor.size(0);
	const auto strel_width = strel_tensor.size(1);
	
	// Output
	const auto batch_size = output_tensor.size(0);
	const auto output_height = output_tensor.size(1);
	const auto output_width = output_tensor.size(2);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Initialize temporal variables 
	float value;
	float candidate;
	int index_i;
	int index_j;
    int index_x;
    int index_y;
	
	// Compute the value of output[k][y][x]
	if (x < output_width && y < output_height) {
		for (int k = 0; k < batch_size; k++) {
			value = -INF;
			for (int j = 0; j < strel_height; j++) {
				for (int i = 0; i < strel_width; i++) {
					candidate = input_tensor[k][y + j][x + i] + strel_tensor[strel_height - (j + 1)][strel_width - (i + 1)];
					if (candidate > value) {
						value = candidate;
						index_i = strel_width - (i + 1);
						index_j = strel_height - (j + 1);
                        index_x = x + i;
                        index_y = y + j;
					}
				}
			}
            output_tensor[k][y][x] = value;
            indexes_strel[k][y][x][0] = index_i;
            indexes_strel[k][y][x][1] = index_j;
            indexes_input[k][y][x][0] = index_x;
            indexes_input[k][y][x][1] = index_y;
		}
	}
}

__global__ void dilation_batched_backward_cuda_kernel(
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output_accessor,
        const torch::PackedTensorAccessor32<short,4,torch::RestrictPtrTraits> indexes_input_accessor,
        const torch::PackedTensorAccessor32<short,4,torch::RestrictPtrTraits> indexes_strel_accessor,
        torch::PackedTensorAccessor32<float,3> grad_input_accessor,
        torch::PackedTensorAccessor32<float,2> grad_strel_accessor,
        const short origin_height,
        const short origin_width) {

    /* Sizes */
	// Size of grad output
	const auto batch_size = grad_output_accessor.size(0);
	const auto grad_output_height = grad_output_accessor.size(1);
	const auto grad_output_width = grad_output_accessor.size(2);
	
	// Compute thread index corresponding in output tensor
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < grad_output_width && y < grad_output_height) {
        // grad_input
        short index_x;
        short index_y;
        for (int k = 0; k < batch_size; k++) {
            index_x = indexes_input_accessor[k][y][x][0];
            index_y = indexes_input_accessor[k][y][x][1];
            if (origin_width <= index_x && index_x < grad_output_width + origin_width &&
                origin_height <= index_y && index_y < grad_output_height + origin_height) {
                atomicAdd(&grad_input_accessor[k][index_y - origin_height][index_x - origin_width],
                          grad_output_accessor[k][y][x]);
            }
        }

        // grad_strel
        short index_i;
        short index_j;
        for (int k = 0; k < batch_size; k++) {
            index_i = indexes_strel_accessor[k][y][x][0];
            index_j = indexes_strel_accessor[k][y][x][1];
            atomicAdd(&grad_strel_accessor[index_j][index_i], grad_output_accessor[k][y][x]);
        }
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

torch::Tensor erosion_dependent_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(1);
	const auto strel_height = strel_tensor.size(2);

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
	auto strel_accessor = strel_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,2>();

	// Launch of the kernel
	erosion_dependent_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor);
	
  	return output_tensor;
}

torch::Tensor dilation_dependent_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_width = input_tensor.size(0);
	const auto input_height = input_tensor.size(1);
	const auto strel_width = strel_tensor.size(1);
	const auto strel_height = strel_tensor.size(2);

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
	auto strel_accessor = strel_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,2>();

	// Launch of the kernel
	dilation_dependent_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor);
	
  	return output_tensor;
}

std::vector<torch::Tensor> erosion_forward_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_height = input_tensor.size(0);
    const auto input_width = input_tensor.size(1);
	const auto strel_height = strel_tensor.size(0);
    const auto strel_width = strel_tensor.size(1);

	const auto output_height = input_height - strel_height + 1;
    const auto output_width = input_width - strel_width + 1;

  	// Initialize output tensor
  	auto options_output = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({output_height, output_width}, options_output);
  	
  	// Initialize indexes
  	auto options_indexes = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt16);
    torch::Tensor indexes_input = torch::zeros({output_height, output_width, 2}, options_indexes);
    torch::Tensor indexes_strel = torch::zeros({output_height, output_width, 2}, options_indexes);

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
    auto indexes_input_accessor = indexes_input.packed_accessor32<short,3>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,3>();

	// Launch of the kernel
	erosion_forward_cuda_kernel<<<grid_size, block_size>>>(
	        input_accessor, strel_accessor, output_accessor, indexes_input_accessor, indexes_strel_accessor);
	
  	return {output_tensor, indexes_input, indexes_strel};
}


std::vector<torch::Tensor> erosion_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape) {

	// Compute output size
	const auto grad_output_height = grad_output.size(0);
    const auto grad_output_width = grad_output.size(1);

	// Recover strel shape
	short* strel_ptr = strel_shape.data_ptr<short>();
  	const short strel_height = strel_ptr[0];
    const short strel_width = strel_ptr[1];

    // Recover origin
    short* origin_ptr = origin_tensor.data_ptr<short>();
    const short origin_height = origin_ptr[0];
    const short origin_width = origin_ptr[1];

  	// Initialize output gradients
  	auto options = torch::TensorOptions().device(grad_output.device());
  	torch::Tensor grad_input = torch::zeros({grad_output_height, grad_output_width}, options);
    torch::Tensor grad_strel = torch::zeros({strel_height, strel_width}, options);
  	
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
	auto indexes_input_accessor = indexes_input.packed_accessor32<short,3,torch::RestrictPtrTraits>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,3,torch::RestrictPtrTraits>();
	auto grad_input_accessor = grad_input.packed_accessor32<float,2>();
    auto grad_strel_accessor = grad_strel.packed_accessor32<float,2>();

	// Launch of the kernel
	erosion_backward_cuda_kernel<<<grid_size, block_size>>>(
	        grad_output_accessor,
	        indexes_input_accessor,
	        indexes_strel_accessor,
	        grad_input_accessor,
	        grad_strel_accessor,
	        origin_height,
	        origin_width);
	
  	return {grad_input, grad_strel};
}

std::vector<torch::Tensor> erosion_batched_forward_cuda(
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
  	auto options_output = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({batch_size, output_height, output_width}, options_output);
  	
  	// Initialize indexes
  	auto options_indexes = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt16);
  	torch::Tensor indexes_input = torch::zeros({batch_size, output_height, output_width, 2}, options_indexes);
    torch::Tensor indexes_strel = torch::zeros({batch_size, output_height, output_width, 2}, options_indexes);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((output_width - 1) / block_width) + 1;
	const int grid_height = ((output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto input_accessor = input_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>();
	auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
	auto output_accessor = output_tensor.packed_accessor32<float,3>();
	auto indexes_input_accessor = indexes_input.packed_accessor32<short,4>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,4>();

	// Launch of the kernel
	erosion_batched_forward_cuda_kernel<<<grid_size, block_size>>>(
	        input_accessor, strel_accessor, output_accessor, indexes_input_accessor, indexes_strel_accessor);

    return {output_tensor, indexes_input, indexes_strel};
}


std::vector<torch::Tensor> erosion_batched_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape) {

	// Compute output size
    const auto batch_size = grad_output.size(0);
	const auto grad_output_height = grad_output.size(1);
	const auto grad_output_width = grad_output.size(2);

    // Recover strel shape
    short* strel_ptr = strel_shape.data_ptr<short>();
    const short strel_height = strel_ptr[0];
    const short strel_width = strel_ptr[1];

    // Recover origin
    short* origin_ptr = origin_tensor.data_ptr<short>();
    const short origin_height = origin_ptr[0];
    const short origin_width = origin_ptr[1];

  	// Initialize output gradients
  	auto options = torch::TensorOptions().device(grad_output.device());
    torch::Tensor grad_input = torch::zeros({batch_size, grad_output_height, grad_output_width}, options);
  	torch::Tensor grad_strel = torch::zeros({strel_height, strel_width}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_height = block_ptr[0];
  	const short block_width = block_ptr[1];
  	
	const int grid_height = ((grad_output_height - 1) / block_height) + 1;
	const int grid_width = ((grad_output_width - 1) / block_width) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

	// Create accessors
	auto grad_output_accessor = grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>();
	auto indexes_input_accessor = indexes_input.packed_accessor32<short,4,torch::RestrictPtrTraits>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,4,torch::RestrictPtrTraits>();
	auto grad_input_accessor = grad_input.packed_accessor32<float,3>();
    auto grad_strel_accessor = grad_strel.packed_accessor32<float,2>();

	// Launch of the kernel
	erosion_batched_backward_cuda_kernel<<<grid_size, block_size>>>(
            grad_output_accessor,
            indexes_input_accessor,
            indexes_strel_accessor,
            grad_input_accessor,
            grad_strel_accessor,
            origin_height,
            origin_width);

    return {grad_input, grad_strel};
}

std::vector<torch::Tensor> dilation_forward_cuda(
    torch::Tensor input_tensor,
    torch::Tensor strel_tensor,
    torch::Tensor block_shape) {

	// Compute output size
	const auto input_height = input_tensor.size(0);
    const auto input_width = input_tensor.size(1);
	const auto strel_height = strel_tensor.size(0);
    const auto strel_width = strel_tensor.size(1);

	const auto output_height = input_height - strel_height + 1;
    const auto output_width = input_width - strel_width + 1;

  	// Initialize output tensor
  	auto options_output = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({output_width, output_height}, options_output);

    // Initialize indexes
    auto options_indexes = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt16);
    torch::Tensor indexes_input = torch::zeros({output_height, output_width, 2}, options_indexes);
    torch::Tensor indexes_strel = torch::zeros({output_height, output_width, 2}, options_indexes);
  	
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
    auto indexes_input_accessor = indexes_input.packed_accessor32<short,3>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,3>();

	// Launch of the kernel
	dilation_forward_cuda_kernel<<<grid_size, block_size>>>(input_accessor, strel_accessor, output_accessor,
                                                            indexes_input_accessor, indexes_strel_accessor);

    return {output_tensor, indexes_input, indexes_strel};
}


std::vector<torch::Tensor> dilation_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape) {

	// Compute output size
	const auto grad_output_height = grad_output.size(0);
    const auto grad_output_width = grad_output.size(1);

    // Recover strel shape
    short* strel_ptr = strel_shape.data_ptr<short>();
    const short strel_height = strel_ptr[0];
    const short strel_width = strel_ptr[1];

    // Recover origin
    short* origin_ptr = origin_tensor.data_ptr<short>();
    const short origin_height = origin_ptr[0];
    const short origin_width = origin_ptr[1];
	
  	// Initialize output gradients
    auto options = torch::TensorOptions().device(grad_output.device());
    torch::Tensor grad_input = torch::zeros({grad_output_height, grad_output_width}, options);
    torch::Tensor grad_strel = torch::zeros({strel_height, strel_width}, options);
  	
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
    auto indexes_input_accessor = indexes_input.packed_accessor32<short,3,torch::RestrictPtrTraits>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,3,torch::RestrictPtrTraits>();
    auto grad_input_accessor = grad_input.packed_accessor32<float,2>();
    auto grad_strel_accessor = grad_strel.packed_accessor32<float,2>();

	// Launch of the kernel
	dilation_backward_cuda_kernel<<<grid_size, block_size>>>(
            grad_output_accessor,
            indexes_input_accessor,
            indexes_strel_accessor,
            grad_input_accessor,
            grad_strel_accessor,
            origin_height,
            origin_width);

    return {grad_input, grad_strel};
}

std::vector<torch::Tensor> dilation_batched_forward_cuda(
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
  	auto options_output = torch::TensorOptions().device(input_tensor.device());
  	torch::Tensor output_tensor = torch::zeros({batch_size, output_height, output_width}, options_output);
  	
  	// Initialize indexes
  	auto options_indexes = torch::TensorOptions().device(input_tensor.device()).dtype(torch::kInt16);
    torch::Tensor indexes_input = torch::zeros({batch_size, output_height, output_width, 2}, options_indexes);
    torch::Tensor indexes_strel = torch::zeros({batch_size, output_height, output_width, 2}, options_indexes);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_width = block_ptr[0];
  	const short block_height = block_ptr[1];
  	
	const int grid_width = ((output_width - 1) / block_width) + 1;
	const int grid_height = ((output_height - 1) / block_height) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

    // Create accessors
    auto input_accessor = input_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    auto strel_accessor = strel_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>();
    auto output_accessor = output_tensor.packed_accessor32<float,3>();
    auto indexes_input_accessor = indexes_input.packed_accessor32<short,4>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,4>();

	// Launch of the kernel
	dilation_batched_forward_cuda_kernel<<<grid_size, block_size>>>(
            input_accessor, strel_accessor, output_accessor, indexes_input_accessor, indexes_strel_accessor);

    return {output_tensor, indexes_input, indexes_strel};
}


std::vector<torch::Tensor> dilation_batched_backward_cuda(
        torch::Tensor grad_output,
        torch::Tensor indexes_input,
        torch::Tensor indexes_strel,
        torch::Tensor strel_shape,
        torch::Tensor origin_tensor,
        torch::Tensor block_shape) {

	// Compute output size
    const auto batch_size = grad_output.size(0);
	const auto grad_output_height = grad_output.size(1);
	const auto grad_output_width = grad_output.size(2);

    // Recover strel shape
    short* strel_ptr = strel_shape.data_ptr<short>();
    const short strel_height = strel_ptr[0];
    const short strel_width = strel_ptr[1];

    // Recover origin
    short* origin_ptr = origin_tensor.data_ptr<short>();
    const short origin_height = origin_ptr[0];
    const short origin_width = origin_ptr[1];

    // Initialize output gradients
    auto options = torch::TensorOptions().device(grad_output.device());
    torch::Tensor grad_input = torch::zeros({batch_size, grad_output_height, grad_output_width}, options);
    torch::Tensor grad_strel = torch::zeros({strel_height, strel_width}, options);
  	
  	// Block & Grid parameters
  	short* block_ptr = block_shape.data_ptr<short>();
  	const short block_height = block_ptr[0];
  	const short block_width = block_ptr[1];
  	
	const int grid_height = ((grad_output_height - 1) / block_height) + 1;
	const int grid_width = ((grad_output_width - 1) / block_width) + 1;
	
	const dim3 block_size(block_width, block_height, 1);
	const dim3 grid_size(grid_width, grid_height, 1);

    // Create accessors
    auto grad_output_accessor = grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>();
    auto indexes_input_accessor = indexes_input.packed_accessor32<short,4,torch::RestrictPtrTraits>();
    auto indexes_strel_accessor = indexes_strel.packed_accessor32<short,4,torch::RestrictPtrTraits>();
    auto grad_input_accessor = grad_input.packed_accessor32<float,3>();
    auto grad_strel_accessor = grad_strel.packed_accessor32<float,2>();

	// Launch of the kernel
	dilation_batched_backward_cuda_kernel<<<grid_size, block_size>>>(
            grad_output_accessor,
            indexes_input_accessor,
            indexes_strel_accessor,
            grad_input_accessor,
            grad_strel_accessor,
            origin_height,
            origin_width);

    return {grad_input, grad_strel};
}
