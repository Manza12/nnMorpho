#include <torch/extension.h>
#include <iostream>
#include <stdio.h>
#include <limits>
#include <stdexcept>

/* Headers */
// Erosion
template <typename scalar>
torch::Tensor erosion(torch::Tensor input, torch::Tensor str_el, torch::Tensor footprint, int origin_x, int origin_y,
                      char border_type, scalar top, scalar bottom, const int block_size_x, const int block_size_y);

// Dilation
template <typename scalar>
torch::Tensor dilation(torch::Tensor input, torch::Tensor str_el, torch::Tensor footprint, int origin_x, int origin_y,
                       scalar bottom, const int block_size_x, const int block_size_y);
