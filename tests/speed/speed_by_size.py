import torch
from time import time
from nnMorpho.greyscale_operators import erosion as greyscale_erosion
from nnMorpho.greyscale_operators import dilation as greyscale_dilation

input_sizes = [(100, 100), (100, 200), (100, 500),
               (1000, 100), (1000, 200), (1000, 500),
               (10000, 100), (1000, 200), (1000, 500)]
str_el_size = (31, 31)

greyscale_operations = [greyscale_erosion, greyscale_dilation]

for input_size in input_sizes:
    print('Input size %s' % str(input_size))
    input_tensor = torch.rand(input_size)
    str_el_tensor = torch.rand(str_el_size)
    footprint = torch.rand(str_el_size) < 0.5
    for operation in greyscale_operations:
        start = time()
        operation(input_tensor, str_el_tensor, footprint)
        elapsed = time() - start
        print('Time to compute %s with size %s: %.3f s' % (operation.__name__, str(input_size), elapsed))
        print('Time per pixel: %.3f μs' % (1e6 * elapsed / (input_size[0] * input_size[1])))
    print()
