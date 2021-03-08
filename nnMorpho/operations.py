import numpy as np
import torch
from torch.nn import functional as f
from typing import Union
import warnings


# Infinite for padding images
INF = 1e6

# Types
NoneType = type(None)
BorderType = Union[int, float, NoneType]


def erosion(image: torch.Tensor, strel: torch.Tensor, origin: tuple = (0, 0), border_value: BorderType = None):
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
        :param strel: torch.Tensor
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
    assert type(strel) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not strel.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    if not (0 <= origin[0] < strel.shape[0] and 0 <= origin[1] < strel.shape[1]):
        raise NotImplementedError('Invalid origin: currently nnMorpho only supports origins within the pixels of the '
                                  'structural element')

    # Fill border value is needed
    if not border_value:
        border_value = INF

    # Convert tensor to float if needed
    if image.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % image.dtype)
        image = image.float()

    # Pad image
    image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1],
                      mode='constant', value=border_value)

    # Unfold the image according to the dimension
    if image.ndim == 2:
        image_extended = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    elif image.ndim == 3:
        image_extended = f.unfold(image_pad.unsqueeze(0), kernel_size=strel.shape)
    elif image.ndim == 4:
        image_extended = f.unfold(image_pad, kernel_size=strel.shape)
    else:
        raise NotImplementedError('Currently nnMorpho only supports 4D tensors of the type (B, C, H, W).')

    # Compute infimum
    strel_flatten = torch.flatten(strel).unsqueeze(0).unsqueeze(-1)
    differences = image_extended - strel_flatten
    result, _ = differences.min(dim=1)

    return torch.reshape(result, image.shape)


def dilation(image: torch.Tensor, strel: torch.Tensor, origin: tuple = (0, 0), border_value: BorderType = None):
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
        :param strel: torch.Tensor
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
    assert type(strel) == torch.Tensor
    assert type(origin) == tuple
    assert type(border_value) in [int, float, NoneType]

    # Check strel dim
    if not strel.ndim == 2:
        raise NotImplementedError('Currently nnMorpho only supports 2D structural elements.')

    # Check origin
    if not (0 <= origin[0] < strel.shape[0] and 0 <= origin[1] < strel.shape[1]):
        raise NotImplementedError(
            'Invalid origin: currently nnMorpho only supports origins within the pixels of the '
            'structural element')

    # Fill border value is needed
    if not border_value:
        border_value = -INF

    # Convert tensor to float if needed
    if image.dtype not in [torch.float32, torch.float64]:
        warnings.warn('Casting image type (%r) to float32 since PyTorch only supports float tensors.' % image.dtype)
        image = image.float()

    # Pad image
    image_pad = f.pad(image, [origin[0], strel.shape[0] - origin[0] - 1, origin[1], strel.shape[1] - origin[1] - 1],
                      mode='constant', value=border_value)

    # Unfold the image according to the dimension
    if image.ndim == 2:
        image_extended = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    elif image.ndim == 3:
        image_extended = f.unfold(image_pad.unsqueeze(0), kernel_size=strel.shape)
    elif image.ndim == 4:
        image_extended = f.unfold(image_pad, kernel_size=strel.shape)
    elif image.ndim == 1:
        image_extended = f.unfold(image_pad.unsqueeze(0).unsqueeze(0).unsqueeze(0), kernel_size=strel.shape)
    else:
        raise NotImplementedError('Currently nnMorpho only supports 4D tensors of the type (B, C, H, W).')

    # Compute supremum
    strel_flip = torch.flip(strel, (0, 1))
    strel_flatten = torch.flatten(strel_flip).unsqueeze(0).unsqueeze(-1)
    sums = image_extended + strel_flatten
    result, _ = sums.max(dim=1)

    return torch.reshape(result, image.shape)


def opening(image: torch.Tensor, strel: torch.Tensor, origin: tuple = (0, 0), border_value: BorderType = None):
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
        :param strel: torch.Tensor
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
    return dilation(erosion(image, strel, origin, border_value), strel, origin, border_value)


def closing(image: torch.Tensor, strel: torch.Tensor, origin: tuple = (0, 0), border_value: BorderType = None):
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
        :param strel: torch.Tensor
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
    return erosion(dilation(image, strel, origin, border_value), strel, origin, border_value)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import configure_logger
    from imageio import imread
    from os.path import join
    from utils import to_greyscale

    log = configure_logger()

    log.info('Running test of basic operations...')

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info('PyTorch device: ' + device.type + ':%r' % device.index)

    _image_tensor = torch.tensor(_image, device=device)
    _strel_tensor = torch.tensor(_strel, device=device)

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
