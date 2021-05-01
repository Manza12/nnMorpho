from nnMorpho.parameters import *


class ErosionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        try:
            input_tensor = args[0]
        except IndexError:
            raise Exception('Insufficient parameters: first argument should be the input tensor.')

        try:
            strel_tensor = args[1]
        except IndexError:
            raise Exception('Insufficient parameters: second argument should be the structural element.')

        try:
            origin = args[2]
        except IndexError:
            raise Exception('Insufficient parameters: third argument should be the origin.')

        try:
            border_value = args[3]
        except IndexError:
            raise Exception('Insufficient parameters: fourth argument should be the border value.')

        input_pad = f.pad(input_tensor,
                          (origin[1], strel_tensor.shape[1] - origin[1] - 1,
                           origin[0], strel_tensor.shape[0] - origin[0] - 1),
                          mode='constant', value=border_value)

        if input_tensor.ndim - strel_tensor.ndim == 0:
            output_tensor, indexes = morphology_cuda.erosion_forward(input_pad, strel_tensor, BLOCK_SHAPE)
        elif input_tensor.ndim - strel_tensor.ndim == 1:
            output_tensor, indexes = morphology_cuda.erosion_batched_forward(input_pad, strel_tensor, BLOCK_SHAPE)
        elif input_tensor.ndim - strel_tensor.ndim == 2:
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("Currently, nnMorpho only supports as input:\n" 
                                      "- 2D tensors of the form (H, W)\n"
                                      "- 3D tensors of the form (B, H, W)"
                                      "- 4D tensors of the form (B, C, H, W)")

        strel_shape = torch.tensor(strel_tensor.shape, dtype=torch.int16)
        ctx.save_for_backward(indexes, strel_shape)

        return output_tensor

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]

        indexes, strel_shape = ctx.saved_tensors
        print(indexes)

        if grad_output.ndim - len(strel_shape) == 0:
            result = morphology_cuda.erosion_backward(grad_output, indexes, strel_shape, BLOCK_SHAPE)
        elif grad_output.ndim - len(strel_shape) == 1:
            result = morphology_cuda.erosion_batched_backward(grad_output, indexes, strel_shape, BLOCK_SHAPE)
        elif grad_output.ndim - len(strel_shape) == 2:
            raise NotImplementedError("ToDo")
        else:
            raise NotImplementedError("Currently, nnMorpho only supports as input:\n" 
                                      "- 2D tensors of the form (H, W)\n"
                                      "- 3D tensors of the form (B, H, W)"
                                      "- 4D tensors of the form (B, C, H, W)")

        return None, result, None, None


class DilationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        input_tensor = args[0]
        strel_tensor = args[1]
        origin = args[2]
        border_value = args[3]

        input_pad = f.pad(input_tensor,
                          (origin[1], strel_tensor.shape[1] - origin[1] - 1,
                           origin[0], strel_tensor.shape[0] - origin[0] - 1),
                          mode='constant', value=border_value)

        output_tensor, indexes = morphology_cuda.dilation_forward(input_pad, strel_tensor, BLOCK_SHAPE)

        strel_shape = torch.tensor(strel_tensor.shape, dtype=torch.int16)
        ctx.save_for_backward(indexes, strel_shape)

        return output_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        indexes, strel_shape = ctx.saved_tensors

        result = morphology_cuda.dilation_backward(grad_output[0], indexes, strel_shape, BLOCK_SHAPE)

        return None, result, None, None


if __name__ == '__main__':
    from imageio import imread
    from os.path import join, isfile
    from os import listdir
    from nnMorpho.utils import to_greyscale
    from nnMorpho.operations import _erosion

    _path = join('..', 'images', 'dataset')
    _images = [im for im in listdir(_path) if isfile(join(_path, im))]

    _images_list = list()
    for im in _images:
        _image = imread(join(_path, im))
        _input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
        _input_tensor = torch.tensor(_input_array)

        _images_list.append(_input_tensor)

    _images_tensor = torch.stack(_images_list, 0)
    _images_tensor = _images_tensor.to(DEVICE)

    _image_tensor = _images_tensor[0]
    _image_batched_tensor = _image_tensor.unsqueeze(0).clone()

    _strel_dim = (7, 7)
    _origin = (_strel_dim[0] // 2, _strel_dim[1] // 2)

    _strel_data = torch.rand(_strel_dim, dtype=torch.float32)
    _strel_tensor_cpu = torch.nn.Parameter(_strel_data, requires_grad=True)
    _strel_tensor_gpu = torch.nn.Parameter(_strel_data.to(DEVICE), requires_grad=True)
    _strel_tensor_batched_cpu = torch.nn.Parameter(_strel_data, requires_grad=True)
    _strel_tensor_batched_gpu = torch.nn.Parameter(_strel_data.to(DEVICE), requires_grad=True)

    _image_eroded_gpu = ErosionFunction.apply(_image_tensor, _strel_tensor_gpu, _origin, INF)
    _image_eroded_cpu = _erosion(_image_tensor.cpu(), _strel_tensor_cpu, _origin, INF)

    _image_batched_eroded_gpu = ErosionFunction.apply(_image_batched_tensor, _strel_tensor_batched_gpu, _origin, INF)
    _image_batched_eroded_cpu = _erosion(_image_batched_tensor.cpu(), _strel_tensor_batched_cpu, _origin, INF)

    error_forward = torch.norm(_image_eroded_gpu.cpu() - _image_eroded_cpu, p=1).item()
    error_forward_batched = torch.norm(_image_batched_eroded_gpu.cpu() - _image_batched_eroded_cpu, p=1).item()
    print("Error forward = ", round(error_forward, 3))
    print("Error forward batched = ", round(error_forward_batched, 3))

    criterion = torch.nn.MSELoss(reduction='mean')

    loss_gpu = criterion(_image_eroded_gpu, torch.zeros_like(_image_eroded_gpu, device=DEVICE))
    loss_cpu = criterion(_image_eroded_cpu, torch.zeros_like(_image_eroded_cpu))
    loss_batched_gpu = criterion(_image_batched_eroded_gpu, torch.zeros_like(_image_batched_eroded_gpu, device=DEVICE))
    loss_batched_cpu = criterion(_image_batched_eroded_cpu, torch.zeros_like(_image_batched_eroded_cpu))

    loss_gpu.backward()
    grad_gpu = _strel_tensor_gpu.grad

    loss_cpu.backward()
    grad_cpu = _strel_tensor_cpu.grad

    loss_batched_gpu.backward()
    grad_batched_gpu = _strel_tensor_batched_gpu.grad

    loss_batched_cpu.backward()
    grad_batched_cpu = _strel_tensor_batched_cpu.grad

    error_backward = torch.norm(grad_gpu.cpu() - grad_cpu, p=1).item()
    error_batched_backward = torch.norm(grad_batched_gpu.cpu() - grad_batched_cpu, p=1).item()
    print("Error backward = ", round(error_backward, 3))
    print("Error batched backward = ", round(error_batched_backward, 3))
