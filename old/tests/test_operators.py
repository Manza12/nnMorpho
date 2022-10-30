import torch
from nnMorpho.greyscale_operators import erosion, dilation
from scipy.ndimage.morphology import grey_erosion


torch_types = [torch.uint8, torch.int8, torch.int16, torch.int, torch.int64,
               torch.float16, torch.float32, torch.float64]
border_types = ['e', 'g']

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

output_erosion_e_list = [
    [7, 2, 3, 4],
    [9, 10, 7, 6],
    [6, 3, 2, 1]
]

output_erosion_g_list = [
    [7, 2, 3, 4],
    [9, 10, 7, 6],
    [6, 3, 2, 1]
]

output_dilation_list = [
    [21, 22, 23, 18],
    [25, 24, 21, 20],
    [20, 19, 26, 15]
]


def test_greyscale_erosion():
    print('Testing erosion...')
    for torch_type in torch_types:
        for border_type in border_types:
            print('Border type %s' % border_type)
            if border_type == 'g':
                print('Input tensor type: %s' % torch_type)
                input_tensor = torch.Tensor(input_list).to(torch_type)
                structuring_element = torch.Tensor(str_el_list).to(torch_type)
                footprint = torch.Tensor(footprint_list).to(torch.bool)
                output_tensor = erosion(input_tensor, structuring_element, footprint, origin=(1, 1), border='geodesic')
                assert torch.sum(torch.abs(output_tensor - torch.Tensor(output_erosion_g_list))) == 0
                print('Output tensor type: %s' % output_tensor.dtype)
                print(output_tensor)
                print()
            else:
                print('Input tensor type: %s' % torch_type)
                input_tensor = torch.Tensor(input_list).to(torch_type)
                structuring_element = torch.Tensor(str_el_list).to(torch_type)
                footprint = torch.Tensor(footprint_list).to(torch.bool)
                output_tensor = erosion(input_tensor, structuring_element, footprint, origin=(1, 1), border='euclidean')
                assert torch.sum(torch.abs(output_tensor - grey_erosion())) == 0
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
        output_tensor = dilation(input_tensor, structuring_element, footprint, origin=(1, 1))
        assert torch.sum(torch.abs(output_tensor - torch.Tensor(output_dilation_list))) == 0
        print('Output tensor type: %s' % output_tensor.dtype)
        print(output_tensor)
        print()


if __name__ == '__main__':
    test_greyscale_erosion()
    test_greyscale_dilation()
