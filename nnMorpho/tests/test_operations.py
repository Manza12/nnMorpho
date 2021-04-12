from scipy.ndimage.morphology import grey_erosion, grey_dilation, grey_opening, grey_closing
from operations import erosion, dilation, opening, closing
import matplotlib.pyplot as plt

from parameters import *


def plot(tensor, title, show=True):
    plt.figure()
    plt.imshow(tensor.cpu().numpy(), cmap='gray', vmin=0, vmax=256)
    plt.title(title)
    if show:
        plt.show()


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
    from utils import to_greyscale

    _path = join('..', '..', 'images')
    _images = [im for im in listdir(_path) if isfile(join(_path, im))]

    # Operations
    _operations = [erosion, dilation, opening, closing]
    _operations_sp = [grey_erosion, grey_dilation, grey_opening, grey_closing]

    # Loop
    for i, _operation in enumerate(_operations):
        print("\n#### Testing", _operation.__name__, "####")
        _operation_sp = _operations_sp[i]
        for im in _images:
            print("\n----\nTreating image", im)

            _image = imread(join(_path, im))
            _input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
            _input_tensor = torch.tensor(_input_array)

            print("Input size:", _input_array.shape)

            plot(_input_tensor, 'Input image', show=False)

            # Scipy
            if _operation == erosion or _operation == opening:
                border_value = INF
            elif _operation == dilation or _operation == closing:
                border_value = -INF
            else:
                raise Exception("Operation unknown")

            print("\nScipy")
            sta = time.time()
            _output_array_scipy = _operation_sp(_input_array, structure=_strel_array, mode='constant', cval=border_value)
            end = time.time()
            print("Time for Scipy:", round(end - sta, 6), "seconds")

            _output_tensor_scipy = torch.tensor(_output_array_scipy)
            plot(_output_tensor_scipy, 'Image after ' + _operation.__name__ + ' - Scipy', show=False)

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
                _output_tensor_cuda = _operation(_input_tensor_cuda, _strel_tensor_cuda, origin=_origin, border_value='geodesic')
                end = time.time()
                time_computation = end - sta
                print("Time for computation:", round(time_computation, 6), "seconds")
                print("Time for nnMorpho:", round(time_computation + time_memory_transfer, 6), "seconds")

                plot(_output_tensor_cuda, 'Image after ' + _operation.__name__ + ' - nnMorpho', show=False)

                error = np.sum(np.abs(_output_tensor_cuda.cpu().numpy() - _output_array_scipy))
                print("Error Scipy/nnMorpho =", error)
            else:
                sta = time.time()
                _output_tensor = erosion(_input_tensor, _strel_tensor, origin=_origin,
                                              border_value='geodesic')
                end = time.time()
                print("Time for nnMorpho:", round(end - sta, 6), "seconds")

                plot(_output_tensor, 'Image after ' + _operation.__name__ + ' - nnMorpho', show=False)

                error = np.sum(np.abs(_output_tensor.numpy() - _output_array_scipy))
                print("Error Scipy/nnMorpho =", error)

    if _show_images:
        plt.show()
