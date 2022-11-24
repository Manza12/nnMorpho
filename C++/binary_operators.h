#include <torch/extension.h>
#include <iostream>
#include <stdio.h>
#include <limits>
#include <stdexcept>

// Erosion
template <typename scalar>
torch::Tensor erosion(torch::Tensor input, torch::Tensor str_el, int origin_x, int origin_y, char border_type,
                      int block_size_x, int block_size_y);

// Dilation
template <typename scalar>
torch::Tensor dilation(torch::Tensor input, torch::Tensor str_el, int origin_x, int origin_y, scalar bottom,
                       int block_size_x, int block_size_y);
