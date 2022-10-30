import torch
from time import time
from nnMorpho.greyscale_operators import erosion as greyscale_erosion
from nnMorpho.greyscale_operators import dilation as greyscale_dilation
from nnMorpho.binary_operators import erosion as binary_erosion
from nnMorpho.binary_operators import dilation as binary_dilation

input_size = (1000, 2000)
str_el_size = (15, 31)

greyscale_operations = [greyscale_erosion, greyscale_dilation]
binary_operations = [binary_erosion, binary_dilation]

torch_types = [torch.uint8, torch.int8, torch.int16, torch.int, torch.int64,
               torch.float16, torch.float32, torch.float64]

for torch_type in torch_types:
    print('Type %s' % torch_type)
    try:
        input_tensor = torch.rand(input_size, dtype=torch_type)
        str_el_tensor = torch.rand(str_el_size, dtype=torch_type)
    except RuntimeError:
        try:
            input_tensor = torch.randint(torch.iinfo(torch_type).max, input_size, dtype=torch_type)
            str_el_tensor = torch.randint(torch.iinfo(torch_type).max, str_el_size, dtype=torch_type)
        except TypeError:
            input_tensor = torch.rand(input_size) < 0.5
            str_el_tensor = torch.rand(str_el_size) < 0.5

    footprint = torch.rand(str_el_size) < 0.5
    for operation in greyscale_operations:
        start = time()
        operation(input_tensor, str_el_tensor, footprint)
        print('Time to compute %s with type %s: %.3f s' % (operation.__name__, torch_type, time() - start))
    print()
