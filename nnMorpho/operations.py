from nnMorpho.parameters import *


def pad_tensor(input_tensor, origin, structural_element, border_value):
    pad_list = []
    for dim in range(structural_element.ndim):
        pad_list += [origin[-dim + 1], structural_element.shape[-dim + 1] - origin[-dim + 1] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)
    return input_pad


def check_parameters(input_tensor, structural_element, origin, border_value):
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


def fill_border(border_value, operation):
    if type(border_value) == str:
        if border_value == 'geodesic':
            if operation == 'erosion':
                border_value = INF
            elif operation == 'dilation':
                border_value = -INF
            else:
                raise ValueError("Invalid operation; should be 'erosion' or 'dilation'")
        elif border_value == 'euclidean':
            border_value = -INF
        else:
            ValueError("Currently string options for border value are: 'geodesic' and 'euclidean'")
    elif type(border_value) in [int, float]:
        pass
    else:
        raise ValueError('The type of the border value should be string, int or float.')

    return border_value


def convert_float(input_tensor, warn=True):
    if not input_tensor.dtype == torch.float32:
        if warn:
            warnings.warn('Casting image type (%r) to float32 since nnMorpho only supports float32 tensors.'
                          % input_tensor.dtype)
        input_tensor = input_tensor.float()

    return input_tensor


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
    return _erosion(input_tensor, structural_element, origin, border_value)


def _erosion(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]],
             border_value: Union[int, float]):
    """ Computation of the erosion
        See :erosion for information about inputs, parameters and outputs.
    """
    # Pad input
    pad_list = [origin[1], structural_element.shape[1] - origin[1] - 1,
                origin[0], structural_element.shape[0] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

    if str(input_tensor.device) == 'cpu':
        # Unfold the input
        input_unfolded = input_pad
        for dim in range(structural_element.ndim):
            input_unfolded = input_unfolded.unfold(dim, structural_element.shape[dim], 1)

        # Differences
        result = input_unfolded - structural_element

        # Take the minimum
        for dim in range(structural_element.ndim):
            result, _ = torch.min(result, dim=-1)
    else:
        result = morphology_cuda.erosion(input_pad, structural_element, BLOCK_SHAPE)

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
    return _dilation(input_tensor, structural_element, origin, border_value)


def _dilation(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]],
              border_value: Union[int, float]):
    """ Computation of the dilation
        See :dilation for information about input, parameters and output.
    """
    # Pad input
    pad_list = [origin[1], structural_element.shape[1] - origin[1] - 1,
                origin[0], structural_element.shape[0] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

    if str(input_tensor.device) == 'cpu':
        # Unfold the input
        input_unfolded = input_pad
        for dim in range(structural_element.ndim):
            input_unfolded = input_unfolded.unfold(dim, structural_element.shape[dim], 1)

        # Sums
        result = input_unfolded + structural_element

        # Take the maximum
        for dim in range(structural_element.ndim):
            result, _ = torch.max(result, dim=-1)
    else:
        result = morphology_cuda.dilation(input_pad, structural_element, BLOCK_SHAPE)

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
    return _opening(input_tensor, structural_element, origin, border_value_erosion, border_value_dilation)


def _opening(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]],
             border_value_erosion: Union[int, float], border_value_dilation: Union[int, float]):
    """ Computation of the opening
            See :opening for information about input, parameters and output.
        """
    return _dilation(_erosion(input_tensor, structural_element, origin, border_value_erosion),
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
    return _closing(input_tensor, structural_element, origin, border_value_dilation, border_value_erosion)


def _closing(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]],
             border_value_dilation: Union[int, float], border_value_erosion: Union[int, float]):
    """ Computation of the closing
            See :closing for information about input, parameters and output.
        """
    return _erosion(_dilation(input_tensor, structural_element, origin, border_value_dilation),
                    structural_element, origin, border_value_erosion)


if __name__ == '__main__':
    # Test Operations
    print("Testing the operations of nnMorpho respect to Scipy")

    # Parameters
    _show_images = True
    _strel_dim = (17, 17)
    _origin = (_strel_dim[0] // 2, _strel_dim[1] // 2)
    _device = 'cuda'
    _device = torch.device("cuda:0" if torch.cuda.is_available() and _device == 'cuda' else "cpu")

    print("\nParameters:")
    print("Showing images:", _show_images)
    print("Structural element dimension:", _strel_dim)
    print("Origin:", _origin)
    print("Device:", _device)

    # Structural element
    _strel_tensor = torch.rand(_strel_dim, dtype=torch.float32) * 12 - 6
    _strel_array = _strel_tensor.numpy()

    # Start CUDA
    if not str(_device) == 'cpu':
        print("\nStarting CUDA")
        sta = time.time()
        _starter = torch.zeros((1, 1), dtype=torch.float32, device=_device)
        end = time.time()
        print("Time for start CUDA:", round(end - sta, 6), "seconds")

    # Inputs
    from imageio import imread
    from os.path import join, isfile
    from os import listdir
    from utils import to_greyscale, plot_image
    from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing
    from matplotlib.pyplot import show

    _path = join('..', 'images')
    _images = [im for im in listdir(_path) if isfile(join(_path, im))]

    # Operations
    _operations = [erosion, dilation, opening, closing]
    _operations_sp = [grey_erosion, grey_dilation, grey_opening, grey_closing]

    # Loop

    for im in _images:
        print("\n----\nTreating image", im)

        _image = imread(join(_path, im))
        _input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
        _input_tensor = torch.tensor(_input_array)

        print("Input size:", _input_array.shape)

        plot_image(_input_tensor, 'Input image', show=False, cmap='gray', v_min=0, v_max=255)

        for i, _operation in enumerate(_operations):
            print("\nTesting", _operation.__name__, "...")
            _operation_sp = _operations_sp[i]

            # Assign border value
            if _operation == erosion or _operation == opening:
                _border_value = INF
            elif _operation == dilation or _operation == closing:
                _border_value = -INF
            else:
                raise Exception("Operation unknown")

            # Scipy
            print("\nScipy")
            sta = time.time()
            _output_array_scipy = _operation_sp(_input_array, structure=_strel_array, mode='constant',
                                                cval=_border_value)
            end = time.time()
            print("Time for Scipy:", round(end - sta, 6), "seconds")

            _output_tensor_scipy = torch.tensor(_output_array_scipy)
            plot_image(_output_tensor_scipy, 'Image after ' + _operation.__name__ + ' - Scipy', show=False, cmap='gray',
                       v_min=0, v_max=255)

            # nnMorpho
            print("\nnnMorpho")

            if not str(_device) == 'cpu':
                # Memory transfer
                sta = time.time()
                _input_tensor_cuda = _input_tensor.to(_device)
                _strel_tensor_cuda = _strel_tensor.to(_device)
                end = time.time()
                time_memory_transfer = end - sta
                print("Time for Memory transfer to GPU:", round(time_memory_transfer, 6), "seconds")

                sta = time.time()
                _output_tensor_cuda = _operation(_input_tensor_cuda, _strel_tensor_cuda, origin=_origin,
                                                 border_value=_border_value)
                end = time.time()
                time_computation = end - sta
                print("Time for computation:", round(time_computation, 6), "seconds")
                print("Time for nnMorpho:", round(time_computation + time_memory_transfer, 6), "seconds")

                plot_image(_output_tensor_cuda, 'Image after ' + _operation.__name__ + ' - nnMorpho', show=False,
                           cmap='gray', v_min=0, v_max=255)

                error = np.matrix.sum(np.abs(_output_tensor_cuda.cpu().numpy() - _output_array_scipy))
                print("Error Scipy/nnMorpho =", error)
            else:
                sta = time.time()
                _output_tensor = erosion(_input_tensor, _strel_tensor, origin=_origin,
                                         border_value='geodesic')
                end = time.time()
                print("Time for nnMorpho:", round(end - sta, 6), "seconds")

                plot_image(_output_tensor, 'Image after ' + _operation.__name__ + ' - nnMorpho', show=False,
                           cmap='gray', v_min=0, v_max=255)

                error = np.matrix.sum(np.abs(_output_tensor.numpy() - _output_array_scipy))
                print("Error Scipy/nnMorpho =", error)

    if _show_images:
        show()
