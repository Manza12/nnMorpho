from parameters import *


def erosion(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
            border_value: BorderType = 'geodesic'):
    """ Erosion is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        erosion of an image by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor that you want to erode. It should be a PyTorch tensor of arbitrary dimension. The
            dimensions that will be eroded are determined by the structural element.
        :param structural_element: torch.Tensor
            The structural element to erode. The structural element should be a PyTorch tensor of arbitrary dimension.
            Its shape should coincide with the shape of the last dimensions of the input_tensor.
        :param origin: tuple, List[int]
            The origin of the structural element. Default to (0, 0). Negative indexes are allowed. No checks are done.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            'geodesic', which puts infinite value in the border to only taking account of points within the input, and
            'euclidean', which puts minus infinite value in the border to only allow full structural element detection.
            Default value is 'geodesic'.

        Outputs
        -------
        :return: torch.Tensor
            The erosion as a PyTorch tensor of the same shape than the original input.
    """
    # Check types
    assert type(input_tensor) == torch.Tensor, 'Input type should be torch.Tensor.'
    assert type(structural_element) == torch.Tensor, 'Structural element type should be torch.Tensor.'
    assert type(origin) in [tuple, List[int]], 'Origin type should be tuple or list[int].'
    assert type(border_value) in [int, float, str], 'Border value type should be int, float or string.'

    # Check dimension of input and structural element are compatible
    assert input_tensor.ndim >= structural_element.ndim, "Input's dimension should be bigger than the structural " \
                                                         "element's one"
    dim_shift = input_tensor.ndim - structural_element.ndim

    # Check origin
    for dim in range(structural_element.ndim):
        assert - input_tensor.shape[dim_shift + dim] < origin[dim] \
               < structural_element.shape[dim] + input_tensor.shape[dim_shift + dim] - 1, \
               'Invalid origin. Structural element and input should intersect at least in one point.'

    # Fill border value if needed
    if type(border_value) == str:
        if border_value == 'geodesic':
            border_value = INF
        elif border_value == 'euclidean':
            border_value = -INF
        else:
            ValueError("Currently string options for border value are: 'geodesic' and 'euclidean'")
    elif type(border_value) == int or type(border_value) == float:
        pass
    else:
        raise ValueError('The type of the border value should be string, int or float.')

    # Convert tensor to float if needed
    if input_tensor.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.'
                      % input_tensor.dtype)
        input_tensor = input_tensor.float()

    return _erosion(input_tensor, structural_element, origin, border_value)


def _erosion(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]],
             border_value: Union[int, float]):
    """ Computation of the erosion
        See :erosion for information about input, parameters and output.
    """
    dim_shift = input_tensor.ndim - structural_element.ndim

    # Pad image
    pad_list = []  # [0] * 2 * dim_shift
    for dim in range(structural_element.ndim):
        pad_list += [origin[-dim+1], structural_element.shape[-dim+1] - origin[-dim+1] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

    # Unfold the input
    input_unfolded = input_pad
    for dim in range(structural_element.ndim):
        input_unfolded = input_unfolded.unfold(dim_shift + dim, structural_element.shape[dim], 1)

    # Differences
    result = input_unfolded - structural_element

    # Take the minimum
    for dim in range(structural_element.ndim):
        result, _ = torch.min(result, dim=-1)

    return result


def dilation(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
             border_value: BorderType = None):
    """ Dilation is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        dilation of an image by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The image that you want to dilate. The image should be a PyTorch tensor with 1, 2, 3 or 4 dimensions. If you
            furnish a 2D tensor, it will be seen as a tensor of the form (H, W) (a grayscale image with height H and
            width W). If you furnish a 3D tensor, it will be seen as a tensor of the form (C, H, W) (a RGB image with
            height H, width W and channels C). If you furnish a 4D tensor, it will be seen as a tensor of the form
            (B, C, H, W) (a batched RGB image where B is the number of batches).
        :param structural_element: torch.Tensor
            The structural element to dilate. The structural element should be a 2D PyTorch tensor.
        :param origin: tuple
            The origin of the structural element. Default to (0, 0). The origin of the structural element.
        :param border_value: int, float
            The value used to pad the image in the border. Default to None. If None, a very low value is chosen such
            that the maximum is only computed inside the image.

        Outputs
        -------
        :return: torch.Tensor
            The dilated image as a PyTorch tensor of the same shape than the original image.
        """
    # Check types
    assert type(input_tensor) == torch.Tensor
    assert type(structural_element) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not structural_element.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    # Todo: change origin limits
    if not (0 <= origin[0] < structural_element.shape[0] and 0 <= origin[1] < structural_element.shape[1]):
        raise NotImplementedError(
            'Invalid origin: currently nnMorpho only supports origins within the pixels of the '
            'structural element')

    # Fill border value if needed
    if not border_value:
        border_value = -INF

    # Convert tensor to float if needed
    if input_tensor.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % input_tensor.dtype)
        input_tensor = input_tensor.float()

    # Compute the dilation
    return _dilation(input_tensor, structural_element, origin, border_value)


def _dilation(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
              border_value: Union[int, float] = -INF):
    """ Computation of the dilation
        See :dilation for information about input, parameters and output.
    """
    # Convert the image to a 4D image of the form (B, C, H, W)
    if input_tensor.ndim == 2:
        image_4d = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.ndim == 3:
        image_4d = input_tensor.unsqueeze(0)
    elif input_tensor.ndim == 4:
        image_4d = input_tensor
    else:
        raise NotImplementedError('Currently nnMorpho only supports image-like tensors.')

    # Pad image
    image_pad = f.pad(image_4d, [origin[0], structural_element.shape[0] - origin[0] - 1, origin[1],
                                 structural_element.shape[1] - origin[1] - 1], mode='constant', value=border_value)

    # Unfold the image
    image_unfolded = f.unfold(image_pad, kernel_size=structural_element.shape)

    # Flatten and flip the structural element
    strel_flatten = torch.flatten(torch.flip(structural_element, (0, 1))).unsqueeze(0).unsqueeze(-1)

    # Sum
    sums = image_unfolded + strel_flatten

    # Take the maximum
    result, _ = torch.max(sums, 1)

    return torch.reshape(result, input_tensor.shape)


def opening(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
            border_value: BorderType = None):
    """ Opening is one of the derived operations of Mathematical Morphology: it consists on eroding an image and then
        dilating it. This function computes the grayscale opening of an image by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The image that you want to open. The image should be a PyTorch tensor with 1, 2, 3 or 4 dimensions. If you
            furnish a 2D tensor, it will be seen as a tensor of the form (H, W) (a grayscale image with height H and
            width W). If you furnish a 3D tensor, it will be seen as a tensor of the form (C, H, W) (a RGB image with
            height H, width W and channels C). If you furnish a 4D tensor, it will be seen as a tensor of the form
            (B, C, H, W) (a batched RGB image where B is the number of batches).
        :param structural_element: torch.Tensor
            The structural element to open. The structural element should be a 2D PyTorch tensor.
        :param origin: tuple
            The origin of the structural element. Default to (0, 0). The origin of the structural element.
        :param border_value: int, float
            The value used to pad the image in the border. Default to None. This argument is passed to the erosion and
            to the dilation.

        Outputs
        -------
        :return: torch.Tensor
            The opened image as a PyTorch tensor of the same shape than the original image.
        """
    # Check types
    assert type(input_tensor) == torch.Tensor
    assert type(structural_element) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not structural_element.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    # Todo: change origin limits
    if not (0 <= origin[0] < structural_element.shape[0] and 0 <= origin[1] < structural_element.shape[1]):
        raise NotImplementedError(
            'Invalid origin: currently nnMorpho only supports origins within the pixels of the '
            'structural element')

    # Fill border value if needed
    if not border_value:
        border_value = -INF

    # Convert tensor to float if needed
    if input_tensor.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % input_tensor.dtype)
        input_tensor = input_tensor.float()

    # Compute the opening
    return _opening(input_tensor, structural_element, origin, border_value)


def _opening(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
             border_value: Union[int, float] = -INF):
    """ Computation of the opening
            See :opening for information about input, parameters and output.
        """
    return _dilation(_erosion(input_tensor, structural_element, origin, border_value),
                     structural_element, origin, border_value)


def closing(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
            border_value: BorderType = None):
    """ Closing is one of the derived operations of Mathematical Morphology: it consists on dilating an image and then
        eroding it. This function computes the grayscale closing of an image by a structural element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The image that you want to close. The image should be a PyTorch tensor with 1, 2, 3 or 4 dimensions. If you
            furnish a 2D tensor, it will be seen as a tensor of the form (H, W) (a grayscale image with height H and
            width W). If you furnish a 3D tensor, it will be seen as a tensor of the form (C, H, W) (a RGB image with
            height H, width W and channels C). If you furnish a 4D tensor, it will be seen as a tensor of the form
            (B, C, H, W) (a batched RGB image where B is the number of batches).
        :param structural_element: torch.Tensor
            The structural element to close. The structural element should be a 2D PyTorch tensor.
        :param origin: tuple
            The origin of the structural element. Default to (0, 0). The origin of the structural element.
        :param border_value: int, float
            The value used to pad the image in the border. Default to None. This argument is passed to the erosion and
            to the dilation.

        Outputs
        -------
        :return: torch.Tensor
            The closed image as a PyTorch tensor of the same shape than the original image.
        """
    # Check types
    assert type(input_tensor) == torch.Tensor
    assert type(structural_element) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not structural_element.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    # Todo: change origin limits
    if not (0 <= origin[0] < structural_element.shape[0] and 0 <= origin[1] < structural_element.shape[1]):
        raise NotImplementedError(
            'Invalid origin: currently nnMorpho only supports origins within the pixels of the '
            'structural element')

    # Fill border value if needed
    if not border_value:
        border_value = -INF

    # Convert tensor to float if needed
    if input_tensor.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % input_tensor.dtype)
        input_tensor = input_tensor.float()

    # Compute the closing
    return _closing(input_tensor, structural_element, origin, border_value)


def _closing(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]] = (0, 0),
             border_value: Union[int, float] = -INF):
    """ Computation of the closing
            See :closing for information about input, parameters and output.
        """
    return _erosion(_dilation(input_tensor, structural_element, origin, border_value),
                    structural_element, origin, border_value)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imageio import imread
    from os.path import join
    # from utils import to_greyscale

    logging.info('Running test of basic operations...')

    _image = imread(join('..', 'images', 'lena.png'))
    # _image = to_greyscale(np.array(_image), warn=False)
    _image = np.array(_image)

    plt.figure()
    # plt.imshow(_image, cmap='gray', vmin=0, vmax=255)
    plt.imshow(_image)
    plt.title('Original image')

    _strel = np.zeros((7, 9))
    _origin = (2, 3)

    _strel_image = np.pad(_strel, ((5, 5), (5, 5)), 'constant', constant_values=-INF)

    plt.figure()
    plt.imshow(_strel_image, cmap='gray', vmin=-100, vmax=0, origin='lower')
    plt.scatter(_origin[0] + 5, _origin[1] + 5, marker='x', c='r')
    plt.title('Structural element\n(red cross is the origin)')

    logging.info('PyTorch device: ' + DEVICE.type + ':%r' % DEVICE.index)

    _image_tensor = torch.tensor(_image, device=DEVICE)
    _image_tensor = _image_tensor.transpose(0, 2).transpose(1, 2)
    _image_tensor = _image_tensor.unsqueeze(0)
    _image_tensor = torch.cat((_image_tensor, _image_tensor), dim=0)

    _strel_tensor = torch.tensor(_strel, device=DEVICE)

    _erosion_tensor = erosion(_image_tensor, _strel_tensor, origin=_origin)

    # First axis
    plt.figure()
    # plt.imshow(_erosion_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    plt.imshow(_erosion_tensor[0, :, :, :].transpose(0, 2).transpose(0, 1).cpu().numpy() / 255)
    plt.title('Erosion')

    # Seconds axis
    plt.figure()
    # plt.imshow(_erosion_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    plt.imshow(_erosion_tensor[1, :, :, :].transpose(0, 2).transpose(0, 1).cpu().numpy() / 255)
    plt.title('Erosion')

    # _dilation_tensor = dilation(_image_tensor, _strel_tensor, origin=_origin)
    # plt.figure()
    # plt.imshow(_dilation_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    # plt.title('Dilation')
    #
    # _opening_tensor = opening(_image_tensor, _strel_tensor, origin=_origin)
    # plt.figure()
    # plt.imshow(_opening_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    # plt.title('Opening')
    #
    # _closing_tensor = closing(_image_tensor, _strel_tensor, origin=_origin)
    # plt.figure()
    # plt.imshow(_closing_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    # plt.title('Closing')

    plt.show()
