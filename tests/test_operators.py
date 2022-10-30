import torch
from nnMorpho.greyscale_operators import erosion as greyscale_erosion
from nnMorpho.greyscale_operators import dilation as greyscale_dilation
from nnMorpho.binary_operators import erosion as binary_erosion
from nnMorpho.binary_operators import dilation as binary_dilation


torch_types = [torch.uint8, torch.int8, torch.int16, torch.int, torch.int64,
               torch.float16, torch.float32, torch.float64]

input_list = [
    [11, 12, 13, 14],
    [19, 16, 15, 12],
    [12, 11, 10, 17]
]

input_binary_list = [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
]

str_el_list = [
    [1, 2, 3],
    [9, 7, 5]
]

str_el_binary_list = [
    [11, 12, 13],
    [10, 12, 12]
]

footprint_list = [
    [1, 1, 1],
    [1, 0, 1]
]

output_erosion_list = [
    [7, 2, 3, 4],
    [9, 10, 7, 6],
    [6, 3, 2, 1]
]

output_dilation_list = [
    [21, 22, 23, 18],
    [25, 24, 21, 20],
    [20, 19, 26, 15]
]

output_binary_erosion_list = [
    [False, True, True, True],
    [False, True, True, True],
    [False, False, False, True]
]

output_binary_dilation_list = [
    [9, 7, 5, 2],
    [1, 2, 9, 7],
    [9, 7, 5, 0]
]


def test_greyscale_erosion():
    print('Testing erosion...')
    for torch_type in torch_types:
        print('Input tensor type: %s' % torch_type)
        input_tensor = torch.Tensor(input_list).to(torch_type)
        structuring_element = torch.Tensor(str_el_list).to(torch_type)
        footprint = torch.Tensor(footprint_list).to(torch.bool)
        output_tensor = greyscale_erosion(input_tensor, structuring_element, footprint, origin=(1, 1), border='geodesic')
        assert torch.all(output_tensor == torch.Tensor(output_erosion_list))
        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()


def test_greyscale_dilation():
    print('Testing dilation...')
    for torch_type in torch_types:
        print('Input tensor type: %s' % torch_type)
        input_tensor = torch.Tensor(input_list).to(torch_type)
        structuring_element = torch.Tensor(str_el_list).to(torch_type)
        footprint = torch.Tensor(footprint_list).to(torch.bool)
        output_tensor = greyscale_dilation(input_tensor, structuring_element, footprint, origin=(1, 1))
        assert torch.all(output_tensor == torch.Tensor(output_dilation_list))
        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()


def test_binary_erosion():
    print('Testing erosion...')
    for torch_type in torch_types:
        print('Input tensor type: %s' % torch_type)
        input_tensor = torch.Tensor(input_list).to(torch_type)
        structuring_element = torch.Tensor(str_el_binary_list).to(torch_type)
        output_tensor = binary_erosion(input_tensor, structuring_element, origin=(1, 1), border='geodesic')
        assert torch.all(output_tensor == torch.Tensor(output_binary_erosion_list))
        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()


def test_binary_dilation():
    print('Testing dilation...')
    for torch_type in torch_types:
        print('Input tensor type: %s' % torch_type)
        input_tensor = torch.Tensor(input_binary_list).to(torch.bool)
        structuring_element = torch.Tensor(str_el_list).to(torch_type)
        output_tensor = binary_dilation(input_tensor, structuring_element, origin=(1, 1))
        output_ground_truth_tensor = torch.Tensor(output_binary_dilation_list)
        try:
            output_ground_truth_tensor[output_ground_truth_tensor == 0] = torch.iinfo(torch_type).min
        except TypeError:
            output_ground_truth_tensor[output_ground_truth_tensor == 0] = -torch.Tensor([float('inf')])
        assert torch.all(output_tensor == output_ground_truth_tensor)
        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()


if __name__ == '__main__':
    test_greyscale_erosion()
    test_greyscale_dilation()
    test_binary_erosion()
    test_binary_dilation()
