from parameters import *


def erosion(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
            border_value: BorderType = None):
    """ Erosion is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        erosion of an image by a structural element.

        Parameters
        ----------
        :param image: torch.Tensor
            The image that you want to erode. The image should be a PyTorch tensor with 1, 2, 3 or 4 dimensions. If you
            furnish a 2D tensor, it will be seen as a tensor of the form (H, W) (a grayscale image with height H and
            width W). If you furnish a 3D tensor, it will be seen as a tensor of the form (C, H, W) (a RGB image with
            height H, width W and channels C). If you furnish a 4D tensor, it will be seen as a tensor of the form
            (B, C, H, W) (a batched RGB image where B is the number of batches).
        :param structural_element: torch.Tensor
            The structural element to erode. The structural element should be a 2D PyTorch tensor.
        :param origin: tuple
            The origin of the structural element. Default to (0, 0). The origin of the structural element.
        :param border_value: int, float
            The value used to pad the image in the border. Default to None. If None, a very big value is chosen such
            that the minimum is only computed inside the image.

        Outputs
        -------
        :return: torch.Tensor
            The eroded image as a PyTorch tensor of the same shape than the original image.

        Note
        ----
        Even if nnMorpho is a general purpose morphology library, it is limited currently by the PyTorch unfold method
        which only allows 4D tensors as input.
    """
    # Check types
    assert type(image) == torch.Tensor
    assert type(structural_element) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not structural_element.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    if not (0 <= origin[0] < structural_element.shape[0] and 0 <= origin[1] < structural_element.shape[1]):
        raise NotImplementedError('Invalid origin: currently nnMorpho only supports origins within the pixels of the '
                                  'structural element')

    # Fill border value if needed
    if not border_value:
        border_value = INF

    # Convert tensor to float if needed
    if image.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % image.dtype)
        image = image.float()

    return _erosion(image, structural_element, origin, border_value)


def _erosion(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
             border_value: Union[int, float] = -INF):
    """ Computation of the erosion
        See :erosion for information about input, parameters and output.
    """
    # Pad image
    image_pad = f.pad(image, [origin[0], structural_element.shape[0] - origin[0] - 1, origin[1],
                              structural_element.shape[1] - origin[1] - 1], mode='constant', value=border_value)

    # Convert the image to a 4D image of the form (B, C, H, W)
    if image.ndim == 2:
        image_4d = image_pad.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image_4d = image_pad.unsqueeze(0)
    elif image.ndim == 4:
        image_4d = image_pad
    else:
        raise NotImplementedError('Currently nnMorpho only supports image-like tensors.')

    # Unfold the image
    image_unfolded = f.unfold(image_4d, kernel_size=structural_element.shape)

    # Flatten the structural element
    strel_flatten = torch.flatten(structural_element).unsqueeze(0).unsqueeze(-1)

    # Differences
    diff = image_unfolded - strel_flatten

    # Take the maximum
    result, _ = torch.min(diff, 1)

    return torch.reshape(result, image.shape)


def dilation(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
             border_value: BorderType = None):
    """ Dilation is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        dilation of an image by a structural element.

        Parameters
        ----------
        :param image: torch.Tensor
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

        Note
        ----
        Even if nnMorpho is a general purpose morphology library, it is limited currently by the PyTorch unfold method
        which only allows 4D tensors as input.
        """
    # Check types
    assert type(image) == torch.Tensor
    assert type(structural_element) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not structural_element.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    if not (0 <= origin[0] < structural_element.shape[0] and 0 <= origin[1] < structural_element.shape[1]):
        raise NotImplementedError(
            'Invalid origin: currently nnMorpho only supports origins within the pixels of the '
            'structural element')

    # Fill border value if needed
    if not border_value:
        border_value = -INF

    # Convert tensor to float if needed
    if image.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % image.dtype)
        image = image.float()

    # Compute the dilation
    return _dilation(image, structural_element, origin, border_value)


def _dilation(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
              border_value: Union[int, float] = -INF):
    """ Computation of the dilation
        See :dilation for information about input, parameters and output.
    """
    # Convert the image to a 4D image of the form (B, C, H, W)
    if image.ndim == 2:
        image_4d = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image_4d = image.unsqueeze(0)
    elif image.ndim == 4:
        image_4d = image
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

    return torch.reshape(result, image.shape)


def opening(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
            border_value: BorderType = None):
    """ Opening is one of the derived operations of Mathematical Morphology: it consists on eroding an image and then
        dilating it. This function computes the grayscale opening of an image by a structural element.

        Parameters
        ----------
        :param image: torch.Tensor
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
    assert type(image) == torch.Tensor
    assert type(structural_element) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not structural_element.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    if not (0 <= origin[0] < structural_element.shape[0] and 0 <= origin[1] < structural_element.shape[1]):
        raise NotImplementedError(
            'Invalid origin: currently nnMorpho only supports origins within the pixels of the '
            'structural element')

    # Fill border value if needed
    if not border_value:
        border_value = -INF

    # Convert tensor to float if needed
    if image.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % image.dtype)
        image = image.float()

    # Compute the opening
    return _opening(image, structural_element, origin, border_value)


def _opening(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
             border_value: Union[int, float] = -INF):
    """ Computation of the opening
            See :opening for information about input, parameters and output.
        """
    return _dilation(_erosion(image, structural_element, origin, border_value),
                     structural_element, origin, border_value)


def closing(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
            border_value: BorderType = None):
    """ Closing is one of the derived operations of Mathematical Morphology: it consists on dilating an image and then
        eroding it. This function computes the grayscale closing of an image by a structural element.

        Parameters
        ----------
        :param image: torch.Tensor
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
    assert type(image) == torch.Tensor
    assert type(structural_element) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not structural_element.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    if not (0 <= origin[0] < structural_element.shape[0] and 0 <= origin[1] < structural_element.shape[1]):
        raise NotImplementedError(
            'Invalid origin: currently nnMorpho only supports origins within the pixels of the '
            'structural element')

    # Fill border value if needed
    if not border_value:
        border_value = -INF

    # Convert tensor to float if needed
    if image.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % image.dtype)
        image = image.float()

    # Compute the closing
    return _closing(image, structural_element, origin, border_value)


def _closing(image: torch.Tensor, structural_element: torch.Tensor, origin: tuple = (0, 0),
             border_value: Union[int, float] = -INF):
    """ Computation of the closing
            See :closing for information about input, parameters and output.
        """
    return _erosion(_dilation(image, structural_element, origin, border_value),
                    structural_element, origin, border_value)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from imageio import imread
    from os.path import join
    from utils import to_greyscale

    logging.info('Running test of basic operations...')

    _image = imread(join('..', 'images', 'lena.png'))
    _image = to_greyscale(np.array(_image), warn=False)

    plt.figure()
    plt.imshow(_image, cmap='gray', vmin=0, vmax=255)
    plt.title('Original image')

    _strel = np.zeros((5, 5))
    _origin = (2, 2)

    _strel_image = np.pad(_strel, ((5, 5), (5, 5)), 'constant', constant_values=-INF)

    plt.figure()
    plt.imshow(_strel_image, cmap='gray', vmin=-100, vmax=0, origin='lower')
    plt.scatter(_origin[0] + 5, _origin[1] + 5, marker='x', c='r')
    plt.title('Structural element\n(red cross is the origin)')

    logging.info('PyTorch device: ' + DEVICE.type + ':%r' % DEVICE.index)

    _image_tensor = torch.tensor(_image, device=DEVICE)
    _strel_tensor = torch.tensor(_strel, device=DEVICE)

    _erosion_tensor = erosion(_image_tensor, _strel_tensor, origin=_origin)
    plt.figure()
    plt.imshow(_erosion_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    plt.title('Erosion')

    _dilation_tensor = dilation(_image_tensor, _strel_tensor, origin=_origin)
    plt.figure()
    plt.imshow(_dilation_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    plt.title('Dilation')

    _opening_tensor = opening(_image_tensor, _strel_tensor, origin=_origin)
    plt.figure()
    plt.imshow(_opening_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    plt.title('Opening')

    _closing_tensor = closing(_image_tensor, _strel_tensor, origin=_origin)
    plt.figure()
    plt.imshow(_closing_tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=255)
    plt.title('Closing')

    plt.show()
