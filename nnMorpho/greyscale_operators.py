import torch
from typing import Optional, Union, List, Tuple
from greyscale_operators_cpp import erosion as erosion_cpp
from greyscale_operators_cpp import dilation as dilation_cpp
from .parameters import BLOCK_SHAPE


def erosion(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            footprint: Optional[torch.Tensor] = None,
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
        :param block_shape: Tuple[int, int]
            The block shape for CUDA computations.

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
    result = erosion_cpp(input_tensor, structuring_element, footprint.to(torch.bool), origin[1], origin[0], border[0],
                         block_shape[1], block_shape[0])

    return result


def dilation(input_tensor: torch.Tensor,
             structuring_element: torch.Tensor,
             footprint: Optional[torch.Tensor] = None,
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
        :param block_shape: Tuple[int, int]
            The block shape for CUDA computations.

        Outputs
        -------
        :return: torch.Tensor
            The dilation as a PyTorch tensor of the same shape as the original input.
        """
    # Adapt origin
    if origin is None:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Create footprint if it does not exist
    if footprint is None:
        footprint = torch.ones_like(structuring_element)

    # Compute erosion
    result = dilation_cpp(input_tensor, structuring_element, footprint.to(torch.bool), origin[1], origin[0],
                          block_shape[1], block_shape[0])

    return result


def opening(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            footprint: Optional[torch.Tensor] = None,
            origin: Optional[Union[tuple, List[int]]] = None,
            border: str = 'e',
            block_shape: Tuple[int, int] = BLOCK_SHAPE):
    """ Opening is one of the derived operations of Mathematical Morphology: it consists on eroding an image and then
        dilating it. This function computes the grayscale opening of an image by a structural element.

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
        :param border: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - [a string starting by 'e']: extends naturally the image setting minus infinite value to the border. ('e'
            stands for Euclidean)
            - [any other value]: only takes into account for taking the minimum the values within the input.
            Default value is 'e'.
        :param block_shape: Tuple[int, int]
            The block shape for CUDA computations.

        Outputs
        -------
        :return: torch.Tensor
            The opening as a PyTorch tensor of the same shape as the original input.
        """

    # Compute opening
    return dilation(erosion(input_tensor, structuring_element, footprint, origin, border, block_shape),
                    structuring_element, footprint, origin, block_shape)


def closing(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            footprint: Optional[torch.Tensor] = None,
            origin: Optional[Union[tuple, List[int]]] = None,
            border: str = 'e',
            block_shape: Tuple[int, int] = BLOCK_SHAPE):
    """ Closing is one of the derived operations of Mathematical Morphology: it consists on dilating an image and then
        eroding it. This function computes the grayscale closing of an image by a structural element.

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
        :param border: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - [a string starting by 'e']: extends naturally the image setting minus infinite value to the border. ('e'
            stands for Euclidean)
            - [any other value]: only takes into account for taking the minimum the values within the input.
            Default value is 'e'.
        :param block_shape: Tuple[int, int]
            The block shape for CUDA computations.

        Outputs
        -------
        :return: torch.Tensor
            The closing as a PyTorch tensor of the same shape as the original input.
        """
    # Compute opening
    return erosion(dilation(input_tensor, structuring_element, footprint, origin, block_shape),
                   structuring_element, footprint, origin, border, block_shape)
