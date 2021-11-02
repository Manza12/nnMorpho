from nnMorpho.operations import erosion, dilation, opening, closing, partial_erosion
from nnMorpho.parameters import *


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

        plot_image(_input_tensor, title='Input image', show=False, cmap='gray', v_min=0, v_max=255)

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
            plot_image(_output_tensor_scipy, title='Image after ' + _operation.__name__ + ' - Scipy', show=False,
                       cmap='gray', v_min=0, v_max=255)

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

                plot_image(_output_tensor_cuda, title='Image after ' + _operation.__name__ + ' - nnMorpho', show=False,
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

                plot_image(_output_tensor, title='Image after ' + _operation.__name__ + ' - nnMorpho', show=False,
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
    _color = True

    print("\nParameters:")
    print("Showing images:", _show_images)
    print("Structural element dimension:", _strel_dim)
    print("Origin:", _origin)
    print("Device:", _device)
    print("Color:", _color)

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

    # Creation of batched images
    print("\nRecovering images...")

    if _color:
        _path = join('..', 'images', 'color')
    else:
        _path = join('..', 'images', 'greyscale')

    _images = [im for im in listdir(_path) if isfile(join(_path, im))]

    _images_list = list()
    _arrays_list = list()
    for im in _images:
        _image = imread(join(_path, im))
        if not _color:
            _input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
            _input_tensor = torch.tensor(_input_array)
        else:
            _input_array = np.array(_image, dtype=np.float32)
            _input_tensor = torch.tensor(_input_array)
            _input_tensor = torch.transpose(_input_tensor, 1, 2)
            _input_tensor = torch.transpose(_input_tensor, 0, 1)

        _images_list.append(_input_tensor)
        _arrays_list.append(_input_array)

    _images_tensor = torch.stack(_images_list, 0)

    # Computations
    print("\nTesting operations...")

    # Scipy
    print("\nScipy")

    _eroded_arrays_list = list()
    _dilated_arrays_list = list()
    _opened_arrays_list = list()
    _closed_arrays_list = list()

    # Erosion
    _border_value = INF
    sta = time.time()
    for im_array in _arrays_list:
        if not _color:
            _output_array_scipy = grey_erosion(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        else:
            _output_array_r = grey_erosion(im_array[:, :, 0], structure=_strel_array, mode='constant',
                                           cval=_border_value)
            _output_array_g = grey_erosion(im_array[:, :, 1], structure=_strel_array, mode='constant',
                                           cval=_border_value)
            _output_array_b = grey_erosion(im_array[:, :, 2], structure=_strel_array, mode='constant',
                                           cval=_border_value)

            _output_array_scipy = np.stack((_output_array_r, _output_array_g, _output_array_b), axis=2)

        _eroded_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_erosion = end - sta
    print("Time for erosion in Scipy:", round(time_scipy_erosion, 6), "seconds")

    _eroded_arrays = np.stack(_eroded_arrays_list, 0)

    # Dilation
    _border_value = -INF
    sta = time.time()
    for im_array in _arrays_list:
        if not _color:
            _output_array_scipy = grey_dilation(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        else:
            _output_array_r = grey_dilation(im_array[:, :, 0], structure=_strel_array, mode='constant',
                                            cval=_border_value)
            _output_array_g = grey_dilation(im_array[:, :, 1], structure=_strel_array, mode='constant',
                                            cval=_border_value)
            _output_array_b = grey_dilation(im_array[:, :, 2], structure=_strel_array, mode='constant',
                                            cval=_border_value)

            _output_array_scipy = np.stack((_output_array_r, _output_array_g, _output_array_b), axis=2)

        _dilated_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_dilation = end - sta
    print("Time for dilation in Scipy:", round(time_scipy_dilation, 6), "seconds")

    _dilated_arrays = np.stack(_dilated_arrays_list, 0)

    # Opening
    _border_value = -INF
    sta = time.time()
    for im_array in _arrays_list:
        if not _color:
            _output_array_scipy = grey_opening(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        else:
            _output_array_r = grey_opening(im_array[:, :, 0], structure=_strel_array, mode='constant',
                                           cval=_border_value)
            _output_array_g = grey_opening(im_array[:, :, 1], structure=_strel_array, mode='constant',
                                           cval=_border_value)
            _output_array_b = grey_opening(im_array[:, :, 2], structure=_strel_array, mode='constant',
                                           cval=_border_value)

            _output_array_scipy = np.stack((_output_array_r, _output_array_g, _output_array_b), axis=2)

        _opened_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_opening = end - sta
    print("Time for opening in Scipy:", round(time_scipy_opening, 6), "seconds")

    _opened_arrays = np.stack(_opened_arrays_list, 0)

    # Closing
    _border_value = -INF
    sta = time.time()
    for im_array in _arrays_list:
        if not _color:
            _output_array_scipy = grey_closing(im_array, structure=_strel_array, mode='constant', cval=_border_value)
        else:
            _output_array_r = grey_closing(im_array[:, :, 0], structure=_strel_array, mode='constant',
                                           cval=_border_value)
            _output_array_g = grey_closing(im_array[:, :, 1], structure=_strel_array, mode='constant',
                                           cval=_border_value)
            _output_array_b = grey_closing(im_array[:, :, 2], structure=_strel_array, mode='constant',
                                           cval=_border_value)

            _output_array_scipy = np.stack((_output_array_r, _output_array_g, _output_array_b), axis=2)

        _closed_arrays_list.append(_output_array_scipy)
    end = time.time()
    time_scipy_closing = end - sta
    print("Time for closing in Scipy:", round(time_scipy_closing, 6), "seconds")

    _closed_arrays = np.stack(_closed_arrays_list, 0)

    # nnMorpho
    print("\nnnMorpho")

    if not str(_device) == 'cpu':
        # Memory transfer
        print("\nMemory transfer")

        sta = time.time()
        _images_tensor_cuda = _images_tensor.to(_device)
        _strel_tensor_cuda = _strel_tensor.to(_device)
        end = time.time()
        time_memory_transfer = end - sta
        print("Time for Memory transfer to GPU:", round(time_memory_transfer, 6), "seconds")

        # Erosion
        print("\nErosion")
        _border_value = INF
        sta = time.time()
        _eroded_images_tensor = erosion(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_erosion = end - sta
        print("Time for computation of the erosion:", round(time_computation_erosion, 6), "seconds")
        time_morpho_erosion = time_computation_erosion + time_memory_transfer
        print("Time for erosion in nnMorpho:", round(time_morpho_erosion, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_eroded_images_tensor - torch.tensor(_eroded_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_eroded_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_eroded_arrays, device=_device), p=1).item()
        print("Error erosion Scipy/nnMorpho =", error)
        print("Improved speed in erosion: x" + str(round(time_scipy_erosion / time_morpho_erosion)))

        # Dilation
        print("\nDilation")
        _border_value = -INF
        sta = time.time()
        _dilated_images_tensor = dilation(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_dilation = end - sta
        print("Time for computation of the dilation:", round(time_computation_dilation, 6), "seconds")
        time_morpho_dilation = time_computation_dilation + time_memory_transfer
        print("Time for dilation in nnMorpho:", round(time_morpho_dilation, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_dilated_images_tensor - torch.tensor(_dilated_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_dilated_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_dilated_arrays, device=_device), p=1).item()
        print("Error dilation Scipy/nnMorpho =", error)
        print("Improved speed in dilation: x" + str(round(time_scipy_dilation / time_morpho_dilation)))

        # Opening
        print("\nOpening")
        _border_value = -INF
        sta = time.time()
        _opened_images_tensor = opening(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_opening = end - sta
        print("Time for computation of the opening:", round(time_computation_opening, 6), "seconds")
        time_morpho_opening = time_computation_opening + time_memory_transfer
        print("Time for opening in nnMorpho:", round(time_morpho_opening, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_opened_images_tensor - torch.tensor(_opened_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_opened_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_opened_arrays, device=_device), p=1).item()
        print("Error opening Scipy/nnMorpho =", error)
        print("Improved speed in opening: x" + str(round(time_scipy_opening / time_morpho_opening)))

        # Closing
        print("\nClosing")
        _border_value = -INF
        sta = time.time()
        _closed_images_tensor = closing(_images_tensor_cuda, _strel_tensor_cuda, _origin, _border_value)
        end = time.time()
        time_computation_closing = end - sta
        print("Time for computation of the closing:", round(time_computation_closing, 6), "seconds")
        time_morpho_closing = time_computation_closing + time_memory_transfer
        print("Time for closing in nnMorpho:", round(time_morpho_closing, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_closed_images_tensor - torch.tensor(_closed_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_closed_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_closed_arrays, device=_device), p=1).item()
        print("Error closing Scipy/nnMorpho =", error)
        print("Improved speed in closing: x" + str(round(time_scipy_closing / time_morpho_closing)))
    else:
        # Erosion
        _border_value = INF
        sta = time.time()
        _eroded_images_tensor = erosion(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for erosion in nnMorpho:", round(end - sta, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_eroded_images_tensor - torch.tensor(_eroded_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_eroded_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_eroded_arrays, device=_device), p=1).item()
        print("Error erosion Scipy/nnMorpho =", error)

        # Dilation
        _border_value = -INF
        sta = time.time()
        _dilated_images_tensor = dilation(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for dilation in nnMorpho:", round(end - sta, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_dilated_images_tensor - torch.tensor(_dilated_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_dilated_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_dilated_arrays, device=_device), p=1).item()
        print("Error dilation Scipy/nnMorpho =", error)

        # Opening
        _border_value = -INF
        sta = time.time()
        _opened_images_tensor = opening(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for opening in nnMorpho:", round(end - sta, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_opened_images_tensor - torch.tensor(_opened_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_opened_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_opened_arrays, device=_device), p=1).item()
        print("Error opening Scipy/nnMorpho =", error)

        # Closing
        _border_value = -INF
        sta = time.time()
        _closed_images_tensor = closing(_images_tensor, _strel_tensor, origin=_origin, border_value=_border_value)
        end = time.time()
        print("Time for closing in nnMorpho:", round(end - sta, 6), "seconds")

        # Error
        if not _color:
            error = torch.norm(_closed_images_tensor - torch.tensor(_closed_arrays, device=_device), p=1).item()
        else:
            error = torch.norm(_closed_images_tensor.transpose(1, 2).transpose(2, 3) -
                               torch.tensor(_closed_arrays, device=_device), p=1).item()
        print("Error closing Scipy/nnMorpho =", error)

    if _show_images:
        for im in range(_images_tensor.shape[0]):
            if not _color:
                plot_four_operations(_images_tensor[im],
                                     _eroded_images_tensor[im],
                                     _dilated_images_tensor[im],
                                     _opened_images_tensor[im],
                                     _closed_images_tensor[im],
                                     title='Image ' + str(im + 1),
                                     show=False, cmap='gray', color=_color, v_min=0, v_max=255)
            else:
                plot_four_operations(_images_tensor[im].transpose(0, 1).transpose(1, 2),
                                     _eroded_images_tensor[im].transpose(0, 1).transpose(1, 2),
                                     _dilated_images_tensor[im].transpose(0, 1).transpose(1, 2),
                                     _opened_images_tensor[im].transpose(0, 1).transpose(1, 2),
                                     _closed_images_tensor[im].transpose(0, 1).transpose(1, 2),
                                     title='Image ' + str(im + 1),
                                     show=False, cmap='gray', color=_color, v_min=0, v_max=255)

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

    plot_image(_input_tensor, title='Input image', show=False, cmap='gray', v_min=0, v_max=255)

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

    plot_image(_output_tensor_cuda, title='Output image - width', show=False, cmap='gray', v_min=0, v_max=255)

    # Partial erosion height
    print("Partial erosion height")
    sta = time.time()
    _output_tensor_cuda = partial_erosion(_input_tensor_cuda.transpose(0, 1), _strel_tensor_cuda, _origin,
                                          _border_value)
    end = time.time()
    time_computation = end - sta
    print("Time for computation:", round(time_computation, 6), "seconds")
    print("Time for partial erosion:", round(time_computation + time_memory_transfer, 6), "seconds")

    plot_image(_output_tensor_cuda.transpose(0, 1), title='Output image - height', show=True, cmap='gray',
               v_min=0, v_max=255)


if __name__ == '__main__':
    test_batched_operations()
    test_common_operations()
    test_partial_erosion()
