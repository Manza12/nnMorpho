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
                result = morphology_cuda.erosion_batched_channel(input_pad, structural_element, BLOCK_SHAPE)
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

    # Compute erosion
    return _partial_erosion(input_tensor, structural_element, origin, border_value)


def _partial_erosion(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]],
                     border_value: Union[int, float]):
    """ Computation of the partial erosion
        See :partial_erosion for information about inputs, parameters and outputs.
    """
    # Pad input
    pad_list = [origin[0], structural_element.shape[1] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)

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
    return _dilation(input_tensor, structural_element, origin, border_value)


def _dilation(input_tensor: torch.Tensor, structural_element: torch.Tensor, origin: Union[tuple, List[int]],
              border_value: Union[int, float]):
    """ Computation of the dilation
        See :dilation for information about input, parameters and output.
    """

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
                result = morphology_cuda.dilation_batched_channel(input_pad, structural_element, BLOCK_SHAPE)
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


def test_common_operations():
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
    from nnMorpho.utils import to_greyscale, plot_image
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
            time_scipy = end - sta
            print("Time for Scipy:", round(time_scipy, 6), "seconds")

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
                time_morpho = time_computation + time_memory_transfer
                print("Time for nnMorpho:", round(time_morpho, 6), "seconds")

                plot_image(_output_tensor_cuda, 'Image after ' + _operation.__name__ + ' - nnMorpho', show=False,
                           cmap='gray', v_min=0, v_max=255)

                error = torch.norm(_output_tensor_cuda - torch.tensor(_output_array_scipy, device=_device), p=1).item()
                print("Error Scipy/nnMorpho =", error)

                if not time_morpho == 0:
                    print("Improved speed: x" + str(round(time_scipy / time_morpho)))
            else:
                sta = time.time()
                _output_tensor = erosion(_input_tensor, _strel_tensor, origin=_origin,
                                         border_value='geodesic')
                end = time.time()
                print("Time for nnMorpho:", round(end - sta, 6), "seconds")

                plot_image(_output_tensor, 'Image after ' + _operation.__name__ + ' - nnMorpho', show=False,
                           cmap='gray', v_min=0, v_max=255)

                error = torch.norm(_output_tensor - torch.tensor(_output_array_scipy), p=1).item()
                print("Error Scipy/nnMorpho =", error)

    if _show_images:
        show()


def test_batched_operations():
    # Test Operations
    print("Testing the batched operations of nnMorpho respect to Scipy")

    # Parameters
    _show_images = True
    _strel_dim = (11, 11)
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
    from nnMorpho.utils import to_greyscale, plot_four_operations
    from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing
    from matplotlib.pyplot import show

    _path = join('..', 'images', 'dataset')
    _images = [im for im in listdir(_path) if isfile(join(_path, im))]

    # Creation of batched images
    print("\nRecovering images...")

    _images_list = list()
    _arrays_list = list()
    for im in _images:
        _image = imread(join(_path, im))
        _input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
        _input_tensor = torch.tensor(_input_array)

        _images_list.append(_input_tensor)
        _arrays_list.append(_input_array)

    _images_tensor = torch.stack(_images_list, 0)

    # Computations
    print("\nTesting operations...")

    # Assign euclidean border value
    _border_value = -INF

    # Scipy
    print("\nScipy")

    _eroded_arrays_list = list()
    _dilated_arrays_list = list()
    _opened_arrays_list = list()
    _closed_arrays_list = list()

    # Erosion
    sta = time.time()
    for im_array in _arrays_list:
        _output_array_scipy = grey_erosion(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        _eroded_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_erosion = end - sta
    print("Time for erosion in Scipy:", round(time_scipy_erosion, 6), "seconds")

    _eroded_arrays = np.stack(_eroded_arrays_list, 0)

    # Dilation
    sta = time.time()
    for im_array in _arrays_list:
        _output_array_scipy = grey_dilation(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        _dilated_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_dilation = end - sta
    print("Time for dilation in Scipy:", round(time_scipy_dilation, 6), "seconds")

    _dilated_arrays = np.stack(_dilated_arrays_list, 0)

    # Opening
    sta = time.time()
    for im_array in _arrays_list:
        _output_array_scipy = grey_opening(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        _opened_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_opening = end - sta
    print("Time for opening in Scipy:", round(time_scipy_opening, 6), "seconds")

    _opened_arrays = np.stack(_opened_arrays_list, 0)

    # Closing
    sta = time.time()
    for im_array in _arrays_list:
        _output_array_scipy = grey_closing(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        _closed_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_closing = end - sta
    print("Time for closing in Scipy:", round(time_scipy_closing, 6), "seconds")

    _closed_arrays = np.stack(_closed_arrays_list, 0)

    # nnMorpho
    print("\nnnMorpho")

    if not str(_device) == 'cpu':
        # Memory transfer
        sta = time.time()
        _images_tensor_cuda = _images_tensor.to(_device)
        _strel_tensor_cuda = _strel_tensor.to(_device)
        end = time.time()
        time_memory_transfer = end - sta
        print("Time for Memory transfer to GPU:", round(time_memory_transfer, 6), "seconds")

        # Erosion
        sta = time.time()
        _eroded_images_tensor = erosion(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_erosion = end - sta
        print("Time for computation of the erosion:", round(time_computation_erosion, 6), "seconds")
        time_morpho_erosion = time_computation_erosion + time_memory_transfer
        print("Time for erosion in nnMorpho:", round(time_morpho_erosion, 6), "seconds")

        # Error
        error = torch.norm(_eroded_images_tensor - torch.tensor(_eroded_arrays, device=_device), p=1).item()
        print("Error erosion Scipy/nnMorpho =", error)
        print("Improved speed in erosion: x" + str(round(time_scipy_erosion / time_morpho_erosion)))

        # Dilation
        sta = time.time()
        _dilated_images_tensor = dilation(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_dilation = end - sta
        print("Time for computation of the dilation:", round(time_computation_dilation, 6), "seconds")
        time_morpho_dilation = time_computation_dilation + time_memory_transfer
        print("Time for dilation in nnMorpho:", round(time_morpho_dilation, 6), "seconds")

        # Error
        error = torch.norm(_dilated_images_tensor - torch.tensor(_dilated_arrays, device=_device), p=1).item()
        print("Error dilation Scipy/nnMorpho =", error)
        print("Improved speed in dilation: x" + str(round(time_scipy_dilation / time_morpho_dilation)))

        # Opening
        sta = time.time()
        _opened_images_tensor = opening(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_opening = end - sta
        print("Time for computation of the opening:", round(time_computation_opening, 6), "seconds")
        time_morpho_opening = time_computation_opening + time_memory_transfer
        print("Time for opening in nnMorpho:", round(time_morpho_opening, 6), "seconds")

        # Error
        error = torch.norm(_opened_images_tensor - torch.tensor(_opened_arrays, device=_device), p=1).item()
        print("Error opening Scipy/nnMorpho =", error)
        print("Improved speed in opening: x" + str(round(time_scipy_opening / time_morpho_opening)))

        # Closing
        sta = time.time()
        _closed_images_tensor = closing(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_closing = end - sta
        print("Time for computation of the closing:", round(time_computation_closing, 6), "seconds")
        time_morpho_closing = time_computation_closing + time_memory_transfer
        print("Time for closing in nnMorpho:", round(time_morpho_closing, 6), "seconds")

        # Error
        error = torch.norm(_closed_images_tensor - torch.tensor(_closed_arrays, device=_device), p=1).item()
        print("Error closing Scipy/nnMorpho =", error)
        print("Improved speed in closing: x" + str(round(time_scipy_closing / time_morpho_closing)))
    else:
        # Erosion
        sta = time.time()
        _eroded_images_tensor = erosion(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for erosion in nnMorpho:", round(end - sta, 6), "seconds")

        error = torch.norm(_eroded_images_tensor - torch.tensor(_eroded_arrays), p=1).item()
        print("Error erosion Scipy/nnMorpho =", error)

        # Dilation
        sta = time.time()
        _dilated_images_tensor = dilation(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for dilation in nnMorpho:", round(end - sta, 6), "seconds")

        error = torch.norm(_dilated_images_tensor - torch.tensor(_dilated_arrays), p=1).item()
        print("Error dilation Scipy/nnMorpho =", error)

        # Opening
        sta = time.time()
        _opened_images_tensor = opening(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for opening in nnMorpho:", round(end - sta, 6), "seconds")

        error = torch.norm(_opened_images_tensor - torch.tensor(_opened_arrays), p=1).item()
        print("Error opening Scipy/nnMorpho =", error)

        # Closing
        sta = time.time()
        _closed_images_tensor = closing(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for closing in nnMorpho:", round(end - sta, 6), "seconds")

        error = torch.norm(_closed_images_tensor - torch.tensor(_closed_arrays), p=1).item()
        print("Error closing Scipy/nnMorpho =", error)

    if _show_images:
        for im in range(_images_tensor.shape[0]):
            plot_four_operations(_images_tensor[im], _eroded_images_tensor[im], _dilated_images_tensor[im],
                                 _opened_images_tensor[im], _closed_images_tensor[im], 'Image ' + str(im + 1),
                                 show=False, cmap='gray',
                                 v_min=0, v_max=255)

    if _show_images:
        show()


def test_partial_erosion():
    # Test partial erosion
    print("Testing the partial erosion")

    # Parameters
    _show_images = True
    _strel_dim = tuple([5])
    _origin = tuple([_strel_dim[0] // 2])
    _border_value = 'geodesic'
    _device = 'cuda'
    _device = torch.device("cuda:0" if torch.cuda.is_available() and _device == 'cuda' else "cpu")

    print("\nParameters:")
    print("Showing images:", _show_images)
    print("Structural element dimension:", _strel_dim)
    print("Origin:", _origin)
    print("Border value:", _border_value)
    print("Device:", _device)

    # Start CUDA
    if not str(_device) == 'cpu':
        print("\nStarting CUDA")
        sta = time.time()
        _starter = torch.zeros((1, 1), dtype=torch.float32, device=_device)
        end = time.time()
        print("Time for start CUDA:", round(end - sta, 6), "seconds")

    # Inputs
    from imageio import imread
    from os.path import join
    from nnMorpho.utils import to_greyscale, plot_image

    _path = join('..', 'images', 'geometry')
    _image = 'vertical_line.png'

    print("\n----\nTreating image", _image)

    _image = imread(join(_path, _image))
    _input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
    _input_tensor = torch.tensor(_input_array)

    print("Input size:", _input_array.shape)

    plot_image(_input_tensor, 'Input image', show=False, cmap='gray', v_min=0, v_max=255)

    # Structural element
    _strel_dim = [_input_array.shape[0], _strel_dim[0]]
    _strel_tensor = torch.zeros(_strel_dim, dtype=torch.float32)
    _strel_array = _strel_tensor.numpy()

    # Memory transfer
    sta = time.time()
    _input_tensor_cuda = _input_tensor.to(_device)
    _strel_tensor_cuda = _strel_tensor.to(_device)
    end = time.time()
    time_memory_transfer = end - sta
    print("Time for Memory transfer to GPU:", round(time_memory_transfer, 6), "seconds")

    # Partial erosion width
    print("Partial erosion width")
    sta = time.time()
    _output_tensor_cuda = partial_erosion(_input_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
    end = time.time()
    time_computation = end - sta
    print("Time for computation:", round(time_computation, 6), "seconds")
    print("Time for partial erosion:", round(time_computation + time_memory_transfer, 6), "seconds")

    plot_image(_output_tensor_cuda, 'Output image - width', show=False, cmap='gray', v_min=0, v_max=255)

    # Partial erosion height
    print("Partial erosion height")
    sta = time.time()
    _output_tensor_cuda = partial_erosion(_input_tensor_cuda.transpose(0, 1), _strel_tensor_cuda, _origin,
                                          _border_value)
    end = time.time()
    time_computation = end - sta
    print("Time for computation:", round(time_computation, 6), "seconds")
    print("Time for partial erosion:", round(time_computation + time_memory_transfer, 6), "seconds")

    plot_image(_output_tensor_cuda.transpose(0, 1), 'Output image - height', show=True, cmap='gray', v_min=0, v_max=255)


if __name__ == '__main__':
    test_batched_operations()
    # test_common_operations()
    # test_partial_erosion()
