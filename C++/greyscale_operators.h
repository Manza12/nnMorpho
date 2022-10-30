#include <torch/extension.h>
#include <iostream>
#include <stdio.h>

// Erosion
template <typename scalar>
torch::Tensor erosion(torch::Tensor input, torch::Tensor str_el, torch::Tensor footprint,
                      int origin_x, int origin_y, char border_type, scalar top, scalar bottom);

// Dilation
template <typename scalar>
torch::Tensor dilation(torch::Tensor input, torch::Tensor str_el, torch::Tensor footprint,
                       int origin_x, int origin_y, scalar top, scalar bottom);
