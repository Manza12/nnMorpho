import torch
from nnMorpho.greyscale_operators import erosion as greyscale_erosion
from nnMorpho.greyscale_operators import dilation as greyscale_dilation
from nnMorpho.binary_operators import erosion as binary_erosion
from nnMorpho.binary_operators import dilation as binary_dilation


torch_types = [torch.uint8, torch.int8, torch.int16, torch.int, torch.int64,
               torch.float16, torch.float32, torch.float64]
devices = ['cpu', 'cuda:0']

input_greyscale_list = [
    [11, 12, 13, 14],
    [19, 16, 15, 12],
    [12, 11, 10, 17]
]

input_binary_erosion_list = [
    [0, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 1, 0, 0]
]

output_binary_dilation_bool_list = [
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [0, 1, 0, 0]
]

str_el_binary_erosion = [
    [0, 1, 1],
    [0, 1, 0]
]

input_binary_dilation_list = [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 1, 0, 0]
]

output_binary_erosion_bool_list = [
    [0, 1, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
]

str_el_greyscale_list = [
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

output_greyscale_erosion_list = [
    [7, 2, 3, 4],
    [9, 10, 7, 6],
    [6, 3, 2, 1]
]

output_greyscale_dilation_list = [
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
    for device in devices:
        print('Device: %s' % device)
        for torch_type in torch_types:
            print('Input tensor type: %s' % torch_type)
            input_tensor = torch.Tensor(input_greyscale_list).to(torch_type).to(device)
            structuring_element = torch.Tensor(str_el_greyscale_list).to(torch_type).to(device)
            footprint = torch.Tensor(footprint_list).to(torch.bool).to(device)
            output_tensor = greyscale_erosion(input_tensor, structuring_element, footprint, origin=(1, 1), border='g')

            ground_truth = torch.Tensor(output_greyscale_erosion_list).to(torch_type).to(device)
            assert torch.all(torch.eq(output_tensor, ground_truth))
            print('Output tensor type: %s' % output_tensor.dtype)
            print(output_tensor)
            print()


def test_greyscale_dilation():
    print('Testing dilation...')
    for device in devices:
        print('Device: %s' % device)
        for torch_type in torch_types:
            print('Input tensor type: %s' % torch_type)
            input_tensor = torch.Tensor(input_greyscale_list).to(torch_type).to(device)
            structuring_element = torch.Tensor(str_el_greyscale_list).to(torch_type).to(device)
            footprint = torch.Tensor(footprint_list).to(torch.bool).to(device)
            output_tensor = greyscale_dilation(input_tensor, structuring_element, footprint, origin=(1, 1))

            ground_truth = torch.Tensor(output_greyscale_dilation_list).to(torch_type).to(device)
            assert torch.all(torch.eq(output_tensor, ground_truth))
            print('Output tensor type: %s' % output_tensor.dtype)
            print(output_tensor)
            print()


def test_binary_erosion():
    print('Testing erosion...')
    for device in devices:
        print('Device: %s' % device)

        # Test bool
        print('Input tensor type: %s' % torch.bool)
        input_tensor = torch.Tensor(input_binary_erosion_list).to(torch.bool).to(device)
        structuring_element = torch.Tensor(str_el_binary_erosion).to(torch.bool).to(device)
        output_tensor = binary_erosion(input_tensor, structuring_element, origin=(1, 1), border='geodesic')

        ground_truth = torch.Tensor(output_binary_erosion_bool_list).to(torch.bool).to(device)
        assert torch.all(torch.eq(output_tensor, ground_truth))
        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()
        # Test other types
        for torch_type in torch_types:
            print('Input tensor type: %s' % torch_type)
            input_tensor = torch.Tensor(input_greyscale_list).to(torch_type).to(device)
            structuring_element = torch.Tensor(str_el_binary_list).to(torch_type).to(device)
            output_tensor = binary_erosion(input_tensor, structuring_element, origin=(1, 1), border='geodesic')

            ground_truth = torch.Tensor(output_binary_erosion_list).to(device)
            assert torch.all(torch.eq(output_tensor, ground_truth))

            print('Output tensor type: %s' % output_tensor.dtype)
            print(output_tensor)
            print()


def test_binary_dilation():
    print('Testing dilation...')
    for device in devices:
        print('Device: %s' % device)

        # Test bool
        print('Input tensor type: %s' % torch.bool)
        input_tensor = torch.Tensor(input_binary_dilation_list).to(torch.bool).to(device)
        structuring_element = torch.Tensor(str_el_binary_erosion).to(torch.bool).to(device)
        output_tensor = binary_dilation(input_tensor, structuring_element, origin=(1, 1))

        ground_truth = torch.Tensor(output_binary_dilation_bool_list).to(device)
        assert torch.all(torch.eq(output_tensor, ground_truth))

        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()
        # Test other types
        for torch_type in torch_types:
            print('Input tensor type: %s' % torch_type)
            input_tensor = torch.Tensor(input_binary_dilation_list).to(torch.bool).to(device)
            structuring_element = torch.Tensor(str_el_greyscale_list).to(torch_type).to(device)
            output_tensor = binary_dilation(input_tensor, structuring_element, origin=(1, 1))

            ground_truth = torch.Tensor(output_binary_dilation_list).to(device)
            try:
                ground_truth[ground_truth == 0] = torch.iinfo(torch_type).min
            except TypeError:
                ground_truth[ground_truth == 0] = -torch.Tensor([float('inf')]).to(device)

            assert torch.all(torch.eq(output_tensor, ground_truth))

            print('Output tensor type: %s' % output_tensor.dtype)
            print(output_tensor)
            print()


if __name__ == '__main__':
    test_greyscale_erosion()
    test_greyscale_dilation()
    test_binary_erosion()
    test_binary_dilation()
