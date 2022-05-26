from nnMorpho.parameters import *
from typing import Tuple


def fill_border(border_value, operation):
    if type(border_value) == str:
        if border_value == 'geodesic':
            if operation == 'erosion':
                border_value = True
            elif operation == 'dilation':
                border_value = False
            else:
                raise ValueError("Invalid operation; should be 'erosion' or 'dilation'")
        elif border_value == 'euclidean':
            border_value = False
        else:
            ValueError("Currently string options for border value are: 'geodesic' and 'euclidean'")
    elif type(border_value) in [int, float]:
        pass
    else:
        raise ValueError('The type of the border value should be string, int or float.')

    return border_value


def erosion(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            origin: Optional[Union[Tuple[int, int], List[int]]] = None,
            border_value: Union[int, float, str] = 'geodesic'
            ):
    # Adapt origin
    if not origin:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Fill border value if needed
    border_value = fill_border(border_value, 'erosion')

    result = cylindric_binary_morphology_cpp.cylindric_binary_erosion(input_tensor, structuring_element,
                                                                      border_value, *origin)

    return result


def dilation(input_tensor: torch.Tensor,
             structuring_element: torch.Tensor,
             origin: Optional[Union[Tuple[int, int], List[int]]] = None,
             border_value: Union[int, float, str] = 'geodesic'
             ):
    # Adapt origin
    if not origin:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Fill border value if needed
    border_value = fill_border(border_value, 'dilation')

    result = cylindric_binary_morphology_cpp.cylindric_binary_dilation(input_tensor, structuring_element,
                                                                       border_value, *origin)

    return result


def hit_or_miss(input_tensor: torch.Tensor,
                structuring_element_in: torch.Tensor,
                structuring_element_out: torch.Tensor,
                origin_in: Optional[tuple] = None,
                origin_out: Optional[tuple] = None,
                border: Optional[str] = 'geodesic'
                ) -> torch.Tensor:
    output_erosion = erosion(input_tensor, structuring_element_in, origin_in, border)
    counter_erosion = erosion(torch.logical_not(input_tensor), structuring_element_out, origin_out, border)
    result = torch.logical_and(output_erosion, counter_erosion)
    return result
