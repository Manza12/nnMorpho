from nnMorpho.parameters import *
from nnMorpho.utils import pad_tensor, fill_border, convert_float
from nnMorpho.checks import check_parameters, check_parameters_partial, check_parameters_dependent


def erosion(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            origin: Optional[Union[tuple, List[int]]] = None,
            border_value: Union[int, float, str] = 'geodesic'):
    """ Erosion is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        erosion of an input tensor by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to erode. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be eroded are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to erode. The structuring element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
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
    check_parameters(input_tensor, structuring_element, origin, border_value)

    # Adapt origin
    if not origin:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Fill border value if needed
    border_value = fill_border(border_value, 'erosion')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Compute erosion
    if str(input_tensor.device) == 'cpu':
        # Pad input
        input_pad = pad_tensor(input_tensor, origin, structuring_element, border_value)

        # Unfold the input
        input_unfolded = input_pad
        dim_shift = input_tensor.ndim - structuring_element.ndim
        for dim in range(structuring_element.ndim):
            input_unfolded = input_unfolded.unfold(dim_shift + dim, structuring_element.shape[dim], 1)

        # Differences
        result = input_unfolded - structuring_element

        # Take the minimum
        for dim in range(structuring_element.ndim):
            result, _ = torch.min(result, dim=-1)
    else:
        if structuring_element.ndim == 2:
            # Pad input
            pad_list = [origin[1], structuring_element.shape[1] - origin[1] - 1,
                        origin[0], structuring_element.shape[0] - origin[0] - 1]
            input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

            if input_tensor.ndim - structuring_element.ndim == 0:
                result = morphology_cuda.erosion(input_pad, structuring_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structuring_element.ndim == 1:
                result = morphology_cuda.erosion_batched(input_pad, structuring_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structuring_element.ndim == 2:
                batch_channel_dim = input_pad.shape[0] * input_pad.shape[1]
                input_height = input_pad.shape[2]
                input_width = input_pad.shape[3]
                input_view = input_pad.view(batch_channel_dim, input_height, input_width)
                result = morphology_cuda.erosion_batched(input_view, structuring_element, BLOCK_SHAPE)
                result = result.view(*input_tensor.shape)
            else:
                raise NotImplementedError("Currently, nnMorpho only supports as input:\n"
                                          "- 2D tensors of the form (H, W)\n"
                                          "- 3D tensors of the form (B, H, W)"
                                          "- 4D tensors of the form (B, C, H, W)")
        else:
            raise NotImplementedError("Currently nnMorpho only supports 2D erosion.")

    return result


def dilation(input_tensor: torch.Tensor,
             structuring_element: torch.Tensor,
             origin: Optional[Union[tuple, List[int]]] = None,
             border_value: Union[int, float, str] = 'geodesic'):
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
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
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
    check_parameters(input_tensor, structuring_element, origin, border_value)

    # Adapt origin
    if not origin:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Fill border value if needed
    border_value = fill_border(border_value, 'dilation')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Compute the dilation
    if str(input_tensor.device) == 'cpu':
        # Pad input
        input_pad = pad_tensor(input_tensor, origin, structuring_element, border_value)

        # Unfold the input
        input_unfolded = input_pad
        dim_shift = input_tensor.ndim - structuring_element.ndim
        for dim in range(structuring_element.ndim):
            input_unfolded = input_unfolded.unfold(dim + dim_shift, structuring_element.shape[dim], 1)

        # Sums
        result = input_unfolded + torch.flip(structuring_element, list(range(structuring_element.ndim)))

        # Take the maximum
        for dim in range(structuring_element.ndim):
            result, _ = torch.max(result, dim=-1)
    else:
        if structuring_element.ndim == 2:
            # Pad input
            pad_list = [origin[1], structuring_element.shape[1] - origin[1] - 1,
                        origin[0], structuring_element.shape[0] - origin[0] - 1]
            input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

            if input_tensor.ndim - structuring_element.ndim == 0:
                result = morphology_cuda.dilation(input_pad, structuring_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structuring_element.ndim == 1:
                result = morphology_cuda.dilation_batched(input_pad, structuring_element, BLOCK_SHAPE)
            elif input_tensor.ndim - structuring_element.ndim == 2:
                batch_channel_dim = input_pad.shape[0] * input_pad.shape[1]
                input_height = input_pad.shape[2]
                input_width = input_pad.shape[3]
                input_view = input_pad.view(batch_channel_dim, input_height, input_width)
                result = morphology_cuda.dilation_batched(input_view, structuring_element, BLOCK_SHAPE)
                result = result.view(*input_tensor.shape)
            else:
                raise NotImplementedError("Currently, nnMorpho only supports as input:\n"
                                          "- 2D tensors of the form (H, W)\n"
                                          "- 3D tensors of the form (B, H, W)"
                                          "- 4D tensors of the form (B, C, H, W)")
        else:
            raise NotImplementedError("Currently nnMorpho only supports 2D erosion.")

    return result


def opening(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            origin: Optional[Union[tuple, List[int]]] = None,
            border_value: Union[int, float, str] = 'geodesic'):
    """ Opening is one of the derived operations of Mathematical Morphology: it consists on eroding an image and then
        dilating it. This function computes the grayscale opening of an image by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to open. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be opened are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to open. The structuring element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
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

    # Compute the opening
    return dilation(erosion(input_tensor, structuring_element, origin, border_value),
                    structuring_element, origin, border_value)


def closing(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            origin: Optional[Union[tuple, List[int]]] = None,
            border_value: Union[int, float, str] = 'geodesic'):
    """ Closing is one of the derived operations of Mathematical Morphology: it consists on dilating an image and then
        eroding it. This function computes the grayscale closing of an image by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to close. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be closed are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to close. The structuring element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
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

    # Compute the closing
    return erosion(dilation(input_tensor, structuring_element, origin, border_value),
                   structuring_element, origin, border_value)


def top_hat(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            origin: Optional[Union[tuple, List[int]]] = None,
            border_value: Union[int, float, str] = 'geodesic'):
    """ Top-hat transform is one of the differential operations of Mathematical Morphology:
        it consists subtracting the opening of an image to the image itself. 
        This function computes the grayscale top-hat of an image by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to transform. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be transformed are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to transform. The structuring element should be a PyTorch tensor of arbitrary
            dimension. Its shape should coincide with the shape of the last dimensions of the input_tensor.
       :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the maximum and the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The top-hat as a PyTorch tensor of the same shape than the original input.
        """

    # Compute the top-hat transform
    return input_tensor - opening(input_tensor, structuring_element, origin, border_value)


def bottom_hat(input_tensor: torch.Tensor,
               structuring_element: torch.Tensor,
               origin: Optional[Union[tuple, List[int]]] = None,
               border_value: Union[int, float, str] = 'geodesic'):
    """ Black Top-hat transform is one of the differential operations of Mathematical Morphology:
        it consists subtracting an image to the closing of the image.
        This function computes the grayscale black top-hat of an image by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to transform. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be transformed are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to transform. The structuring element should be a PyTorch tensor of arbitrary
            dimension. Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the maximum and the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The black top-hat as a PyTorch tensor of the same shape than the original input.
        """

    # Compute the black top-hat transform
    return closing(input_tensor, structuring_element, origin, border_value) - input_tensor


white_top_hat = top_hat
black_top_hat = bottom_hat


def internal_gradient(input_tensor: torch.Tensor,
                      structuring_element: torch.Tensor,
                      origin: Optional[Union[tuple, List[int]]] = None,
                      border_value: Union[int, float, str] = 'geodesic'):
    """ Internal gradient is one of the differential operations of Mathematical Morphology:
        it consists subtracting the erosion of an image to the image itself.
        This function computes the internal gradient of an image by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to transform. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be transformed are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to transform. The structuring element should be a PyTorch tensor of arbitrary
            dimension. Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the maximum and the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The internal gradient as a PyTorch tensor of the same shape than the original input.
        """

    # Compute the internal gradient
    return input_tensor - erosion(input_tensor, structuring_element, origin, border_value)


def external_gradient(input_tensor: torch.Tensor,
                      structuring_element: torch.Tensor,
                      origin: Optional[Union[tuple, List[int]]] = None,
                      border_value: Union[int, float, str] = 'geodesic'):
    """ External gradient is one of the differential operations of Mathematical Morphology:
        it consists subtracting an image to the dilation of the image.
        This function computes the external gradient of an image by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to transform. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be transformed are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to transform. The structuring element should be a PyTorch tensor of arbitrary
            dimension. Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the maximum and the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The external gradient as a PyTorch tensor of the same shape than the original input.
        """

    # Compute the internal gradient
    return dilation(input_tensor, structuring_element, origin, border_value) - input_tensor


def gradient(input_tensor: torch.Tensor,
             structuring_element: torch.Tensor,
             origin: Optional[Union[tuple, List[int]]] = None,
             border_value: Union[int, float, str] = 'geodesic'):
    """ Gradient is one of the differential operations of Mathematical Morphology:
        it consists subtracting the erosion of an image to the dilation of the image.
        This function computes the gradient of an image by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to transform. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be transformed are determined by the structuring element.
        :param structuring_element: torch.Tensor
            The structuring element to transform. The structuring element should be a PyTorch tensor of arbitrary
            dimension. Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the maximum and the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The gradient as a PyTorch tensor of the same shape than the original input.
        """

    # Compute the internal gradient
    return dilation(input_tensor, structuring_element, origin,
                    border_value) - erosion(input_tensor, structuring_element, origin, border_value)


def erosion_dependent(input_tensor: torch.Tensor,
                      structuring_element: torch.Tensor,
                      origin: Optional[Union[tuple, List[int]]] = None,
                      border_value: Union[int, float, str] = 'geodesic'):
    """ This type of erosion is needed when you want a structuring element to vary along one axis.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to erode. It should be a PyTorch tensor of 2 dimensions.
        :param structuring_element: torch.Tensor
            The structuring element to erode. The structuring element should be a PyTorch tensor of 3 dimensions;
            first dimension should coincide with first dimension of input_tensor and two other dimensions are the
            shape of the structuring element.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed. The origin will be the same for all the structuring elements.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The erosion dependent of the first axis as a PyTorch tensor of the same shape than the original input.
    """
    # Check parameters
    check_parameters_dependent(input_tensor, structuring_element, origin, border_value)

    # Adapt origin
    if not origin:
        origin = (structuring_element.shape[1] // 2, structuring_element.shape[2] // 2)

    # Fill border value if needed
    border_value = fill_border(border_value, 'erosion')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Pad input
    pad_list = [origin[1], structuring_element.shape[2] - origin[1] - 1,
                origin[0], structuring_element.shape[1] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

    # Compute erosion
    if str(input_tensor.device) == 'cpu':
        raise ValueError('Operation currently only implemented for GPU.')
    else:
        result = morphology_cuda.erosion_dependent(input_pad, structuring_element, BLOCK_SHAPE)

    return result


def dilation_dependent(input_tensor: torch.Tensor,
                       structuring_element: torch.Tensor,
                       origin: Optional[Union[tuple, List[int]]] = None,
                       border_value: Union[int, float, str] = 'geodesic'):
    """ This type of dilation is needed when you want a structuring element to vary along one axis.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to dilate. It should be a PyTorch tensor of 2 dimensions.
        :param structuring_element: torch.Tensor
            The structuring element to dilate. The structuring element should be a PyTorch tensor of 3 dimensions;
            first dimension should coincide with first dimension of input_tensor and two other dimensions are the
            shape of the structuring element.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed. The origin will be the same for all the structuring elements.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The dilation dependent of the first axis as a PyTorch tensor of the same shape than the original input.
    """
    # Check parameters
    check_parameters_dependent(input_tensor, structuring_element, origin, border_value)

    # Adapt origin
    if not origin:
        origin = (structuring_element.shape[1] // 2, structuring_element.shape[2] // 2)

    # Fill border value if needed
    border_value = fill_border(border_value, 'dilation')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Pad input
    pad_list = [origin[1], structuring_element.shape[2] - origin[1] - 1,
                origin[0], structuring_element.shape[1] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

    # Compute dilation
    if str(input_tensor.device) == 'cpu':
        raise ValueError('Operation currently only implemented for GPU.')
    else:
        result = morphology_cuda.dilation_dependent(input_pad, structuring_element, BLOCK_SHAPE)

    return result


def partial_erosion(input_tensor: torch.Tensor,
                    structuring_element: torch.Tensor,
                    origin: Optional[Union[tuple, List[int]]] = None,
                    border_value: Union[int, float, str] = 'geodesic'):
    # ToDo: Improve the documentation
    """ Partial erosion is a new operation that does a one-dimension-long erosion.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
        :param structuring_element: torch.Tensor
        :param origin: tuple, List[int]
        :param border_value: int, float, str

        Outputs
        -------
        :return: torch.Tensor
    """
    # Check parameters
    check_parameters_partial(input_tensor, structuring_element, origin, border_value)

    # Adapt origin
    if not origin:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[0] // 2)

    # Fill border value if needed
    border_value = fill_border(border_value, 'erosion')

    # Convert tensor to float if needed
    input_tensor = convert_float(input_tensor)

    # Pad input
    pad_list = [origin[0], structuring_element.shape[1] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

    # Compute erosion
    if str(input_tensor.device) == 'cpu':
        raise NotImplementedError("CPU computation is not implemented yet for partial erosion.")
    else:
        result = morphology_cuda.partial_erosion(input_pad, structuring_element, BLOCK_SHAPE)

    return result
