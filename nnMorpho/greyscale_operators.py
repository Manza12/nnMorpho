import torch
from torch import Tensor
from typing import Optional, Union, List, Tuple
from greyscale_operators_cpp import erosion as erosion_cpp
from greyscale_operators_cpp import dilation as dilation_cpp


BLOCK_SHAPE = (32, 32)


def erosion(input_tensor: Tensor,
            structuring_element: Tensor,
            footprint: Optional[Tensor] = None,
            origin: Optional[Union[tuple, List[int]]] = None,
            border: str = 'e',
            block_shape: Tuple[int, int] = BLOCK_SHAPE):
    """ Erosion is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        erosion of an input tensor by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor to erode. It should be a 2D PyTorch tensor.
        :param structuring_element: torch.Tensor
            The structuring element to erode. The structuring element should be a 2D PyTorch tensor.
        :param footprint: torch.Tensor
            The footprint to erode. The footprint should be a PyTorch tensor with the same dimension as the structuring
            element.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
        :param border: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - [a string starting by 'e']: extends naturally the image setting minus infinite value to the border. ('e'
            stands for Euclidean)
            - [any other value]: only takes into account for taking the minimum the values within the input.
            Default value is 'e'.

        Outputs
        -------
        :return: torch.Tensor
            The erosion as a PyTorch tensor of the same shape of the original input.
    """
    # Adapt origin
    if origin is None:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Create footprint if it does not exist
    if footprint is None:
        footprint = torch.ones_like(structuring_element)

    # Compute erosion
    result = erosion_cpp(input_tensor, structuring_element, footprint.to(torch.bool), origin[0], origin[1], border[0],
                         block_shape[0], block_shape[1])

    return result


def dilation(input_tensor: Tensor,
             structuring_element: Tensor,
             footprint: Optional[Tensor] = None,
             origin: Optional[Union[tuple, List[int]]] = None,
             block_shape: Tuple[int, int] = BLOCK_SHAPE):
    """ Dilation is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        dilation of an input tensor by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to dilate. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be dilated are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to dilate. The structuring element should be a PyTorch tensor of arbitrary
            dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param footprint: torch.Tensor
            The footprint to erode. The footprint should be a PyTorch tensor with the same dimension as the structuring
            element.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.

        Outputs
        -------
        :return: torch.Tensor
            The dilation as a PyTorch tensor of the same shape than the original input.
        """
    # Adapt origin
    if origin is None:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Create footprint if it does not exist
    if footprint is None:
        footprint = torch.ones_like(structuring_element)

    # Compute erosion
    result = dilation_cpp(input_tensor, structuring_element, footprint.to(torch.bool), origin[0], origin[1],
                          block_shape[0], block_shape[1])

    return result
