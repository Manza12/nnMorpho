from nnMorpho.parameters import *


def check_parameters(input_tensor, structural_element, origin, border_value):
    # Check types
    assert type(input_tensor) == torch.Tensor, 'Input type should be torch.Tensor.'
    assert type(structural_element) == torch.Tensor, 'Structural element type should be torch.Tensor.'
    assert type(origin) in [tuple, List[int], type(None)], 'Origin type should be None, tuple or list[int].'
    assert type(border_value) in [int, float, str], 'Border value type should be int, float or string.'

    # Check dimension of input and structural element are compatible and compatible with the origin
    assert input_tensor.ndim >= structural_element.ndim, "Input's dimension should be bigger than the structural " \
                                                         "element's one"
    assert structural_element.ndim == len(origin), "The length of the origin should be the same as the number of " \
                                                   "dimensions of the structural element."
    dim_shift = input_tensor.ndim - structural_element.ndim

    # Check origin
    for dim in range(structural_element.ndim):
        assert - input_tensor.shape[dim_shift + dim] < origin[dim] \
               < structural_element.shape[dim] + input_tensor.shape[dim_shift + dim] - 1, \
               'Invalid origin. Structural element and input should intersect at least in one point.'


def check_parameters_partial(input_tensor, structural_element, origin, border_value):
    # Check types
    assert type(input_tensor) == torch.Tensor, 'Input type should be torch.Tensor.'
    assert type(structural_element) == torch.Tensor, 'Structural element type should be torch.Tensor.'
    assert type(origin) in [tuple, List[int]], 'Origin type should be tuple or list[int].'
    assert type(border_value) in [int, float, str], 'Border value type should be int, float or string.'

    # Check dimension of input and structural element are compatible
    assert input_tensor.ndim == structural_element.ndim, "Input's dimension should be the same as the structural " \
                                                         "element's ones"
    assert input_tensor.shape[0] == structural_element.shape[0], "First dimension should coincide between input and " \
                                                                 "structural element."

    # Check origin
    assert len(origin) == 1, "Only origin for the second dimension is needed."
    assert - input_tensor.shape[1] < origin[0] < structural_element.shape[1] + input_tensor.shape[1] - 1, \
        'Invalid origin. Structural element and input should intersect at least in one point.'
