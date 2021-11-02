from nnMorpho.parameters import *
from nnMorpho.utils import pad_tensor, fill_border, convert_float
from nnMorpho.checks import check_parameters, check_parameters_partial


def erosion(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
            border_value: Union[int, float, str] = 'geodesic'):
    """ Erosion is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        erosion of an input tensor by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to erode. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be eroded are determined by the structural element.
        :param structural_element: torch.Tensor
            The structural element to erode. The structural element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: tuple, List[int]
            The origin of the structural element. Default to (0, 0). Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The erosion as a PyTorch tensor of the same shape than the original input.
    """
    # Check parameters
    check_parameters(input_tensor, structural_element, origin, border_value)

    # Fill border value if needed
    border_value = fill_border(border_value, 'erosion')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Compute erosion
    if str(input_tensor.device) == 'cpu':
        # Pad input
        input_pad = pad_tensor(input_tensor, origin, structural_element, border_value)

        # Unfold the input
        input_unfolded = input_pad
        dim_shift = input_tensor.ndim - structural_element.ndim
        for dim in range(structural_element.ndim):
            input_unfolded = input_unfolded.unfold(dim_shift + dim, structural_element.shape[dim], 1)

        # Differences
        result = input_unfolded - structural_element

        # Take the minimum
        for dim in range(structural_element.ndim):
            result, _ = torch.min(result, dim=-1)
    else:
        if structural_element.ndim == 2:
            # Pad input
            pad_list = [origin[1], structural_element.shape[1] - origin[1] - 1,
                        origin[0], structural_element.shape[0] - origin[0] - 1]
            input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

            if input_tensor.ndim - structural_element.ndim == 0:
                result = morphology_cuda.erosion(input_pad, structural_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structural_element.ndim == 1:
                result = morphology_cuda.erosion_batched(input_pad, structural_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structural_element.ndim == 2:
                batch_channel_dim = input_pad.shape[0] * input_pad.shape[1]
                input_height = input_pad.shape[2]
                input_width = input_pad.shape[3]
                input_view = input_pad.view(batch_channel_dim, input_height, input_width)
                result = morphology_cuda.erosion_batched(input_view, structural_element, BLOCK_SHAPE)
                result = result.view(*input_tensor.shape)
            else:
                raise NotImplementedError("Currently, nnMorpho only supports as input:\n"
                                          "- 2D tensors of the form (H, W)\n"
                                          "- 3D tensors of the form (B, H, W)"
                                          "- 4D tensors of the form (B, C, H, W)")
        else:
            raise NotImplementedError("Currently nnMorpho only supports 2D erosion.")

    return result


def partial_erosion(input_tensor: torch.Tensor, structural_element: torch.Tensor,
                    origin: Union[tuple, List[int]] = (0, 0), border_value: Union[int, float, str] = 'geodesic'):
    # ToDo: Improve the documentation
    """ Partial erosion is a new operation that does a one-dimension-long erosion.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
        :param structural_element: torch.Tensor
        :param origin: tuple, List[int]
        :param border_value: int, float, str

        Outputs
        -------
        :return: torch.Tensor
    """
    # Check parameters
    check_parameters_partial(input_tensor, structural_element, origin, border_value)

    # Fill border value if needed
    border_value = fill_border(border_value, 'erosion')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Pad input
    pad_list = [origin[0], structural_element.shape[1] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

    # Compute erosion
    if str(input_tensor.device) == 'cpu':
        raise NotImplementedError("CPU computation is not implemented yet for partial erosion.")
    else:
        result = morphology_cuda.partial_erosion(input_pad, structural_element, BLOCK_SHAPE)

    return result


def dilation(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
             border_value: Union[int, float, str] = 'geodesic'):
    """ Dilation is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        dilation of an input tensor by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to dilate. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be dilated are determined by the structural element.
        :param structural_element: torch.Tensor
            The structural element to dilate. The structural element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: tuple, List[int]
            The origin of the structural element. Default to (0, 0). Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the maximum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The dilation as a PyTorch tensor of the same shape than the original input.
        """
    # Check parameters
    check_parameters(input_tensor, structural_element, origin, border_value)

    # Fill border value if needed
    border_value = fill_border(border_value, 'dilation')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Compute the dilation
    if str(input_tensor.device) == 'cpu':
        # Pad input
        input_pad = pad_tensor(input_tensor, origin, structural_element, border_value)

        # Unfold the input
        input_unfolded = input_pad
        dim_shift = input_tensor.ndim - structural_element.ndim
        for dim in range(structural_element.ndim):
            input_unfolded = input_unfolded.unfold(dim + dim_shift, structural_element.shape[dim], 1)

        # Sums
        result = input_unfolded + torch.flip(structural_element, list(range(structural_element.ndim)))

        # Take the maximum
        for dim in range(structural_element.ndim):
            result, _ = torch.max(result, dim=-1)
    else:
        if structural_element.ndim == 2:
            # Pad input
            pad_list = [origin[1], structural_element.shape[1] - origin[1] - 1,
                        origin[0], structural_element.shape[0] - origin[0] - 1]
            input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

            if input_tensor.ndim - structural_element.ndim == 0:
                result = morphology_cuda.dilation(input_pad, structural_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structural_element.ndim == 1:
                result = morphology_cuda.dilation_batched(input_pad, structural_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structural_element.ndim == 2:
                batch_channel_dim = input_pad.shape[0] * input_pad.shape[1]
                input_height = input_pad.shape[2]
                input_width = input_pad.shape[3]
                input_view = input_pad.view(batch_channel_dim, input_height, input_width)
                result = morphology_cuda.dilation_batched(input_view, structural_element, BLOCK_SHAPE)
                result = result.view(*input_tensor.shape)
            else:
                raise NotImplementedError("Currently, nnMorpho only supports as input:\n"
                                          "- 2D tensors of the form (H, W)\n"
                                          "- 3D tensors of the form (B, H, W)"
                                          "- 4D tensors of the form (B, C, H, W)")
        else:
            raise NotImplementedError("Currently nnMorpho only supports 2D erosion.")

    return result


def opening(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
            border_value: Union[int, float, str] = 'geodesic'):
    """ Opening is one of the derived operations of Mathematical Morphology: it consists on eroding an image and then
        dilating it. This function computes the grayscale opening of an image by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to open. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be opened are determined by the structural element.
        :param structural_element: torch.Tensor
            The structural element to open. The structural element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: tuple, List[int]
            The origin of the structural element. Default to (0, 0). Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the minimum and the maximum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The opening as a PyTorch tensor of the same shape than the original input.
        """
    # Check parameters
    check_parameters(input_tensor, structural_element, origin, border_value)

    # Fill border value if needed
    border_value_erosion = fill_border(border_value, 'erosion')
    border_value_dilation = fill_border(border_value, 'dilation')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Compute the opening
    return dilation(erosion(input_tensor, structural_element, origin, border_value_erosion),
                    structural_element, origin, border_value_dilation)


def closing(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
            border_value: Union[int, float, str] = 'geodesic'):
    """ Closing is one of the derived operations of Mathematical Morphology: it consists on dilating an image and then
        eroding it. This function computes the grayscale closing of an image by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to close. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be closed are determined by the structural element.
        :param structural_element: torch.Tensor
            The structural element to close. The structural element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: tuple, List[int]
            The origin of the structural element. Default to (0, 0). Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the maximum and the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The closing as a PyTorch tensor of the same shape than the original input.
        """
    # Check parameters
    check_parameters(input_tensor, structural_element, origin, border_value)

    # Fill border value if needed
    border_value_erosion = fill_border(border_value, 'erosion')
    border_value_dilation = fill_border(border_value, 'dilation')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Compute the closing
    return erosion(dilation(input_tensor, structural_element, origin, border_value_dilation),
                   structural_element, origin, border_value_erosion)
