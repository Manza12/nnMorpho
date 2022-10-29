import torch
from nnMorpho.greyscale_operators import erosion


torch_types = [torch.uint8, torch.int8, torch.int16, torch.int, torch.int64,
               torch.float16, torch.float32, torch.float64]

input_list = [
    [11, 12, 13, 14],
    [19, 16, 15, 12],
    [12, 11, 10, 17]
]

str_el_list = [
    [1, 2, 3],
    [9, 7, 5]
]

footprint_list = [
    [1, 1, 1],
    [1, 0, 1]
]

output_list = [
    [7, 2, 3, 4],
    [9, 10, 7, 6],
    [6, 3, 2, 1]
]


def test_greyscale_erosion():
    for torch_type in torch_types:
        print('Input tensor type: %s' % torch_type)
        input_tensor = torch.Tensor(input_list).to(torch_type)
        structuring_element = torch.Tensor(str_el_list).to(torch_type)
        footprint = torch.Tensor(footprint_list).to(torch.bool)
        output_tensor = erosion(input_tensor, structuring_element, footprint, origin=(1, 1), border_value='geodesic')
        assert torch.sum(torch.abs(output_tensor - torch.Tensor(output_list))) == 0
        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()


if __name__ == '__main__':
    test_greyscale_erosion()
