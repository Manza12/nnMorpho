from functions import ErosionPythonFunction, ErosionFunction
from imageio import imread
from os.path import join
from utils import to_greyscale
import numpy as np
import torch
import torch.nn.functional as f
from operations import erosion

_image = imread(join('..', '..', 'images', 'test.png'))
_input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
_input_tensor = torch.tensor(_input_array, device='cuda:0')
_strel_target = torch.tensor([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]], device='cuda:0')
_target_image = erosion(_input_tensor, _strel_target, (1, 1), 0)
_strel_tensor = torch.rand((3, 3), device='cuda:0')
# _strel_tensor[0, 0] = -1
# _strel_tensor[0, 1] = -1
_strel_tensor.requires_grad = True
_strel_tensor_old = _strel_tensor.detach().clone()
_strel_tensor_python = _strel_tensor.detach().clone()
_strel_tensor_old.requires_grad = True
_strel_tensor_python.requires_grad = True


def old_erosion(input_tensor, strel, origin, border_value):
    pad_list = [origin[1], strel.shape[1] - origin[1] - 1,
                origin[0], strel.shape[0] - origin[0] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)
    # Unfold the input
    input_unfolded = input_pad
    for dim in range(strel.ndim):
        input_unfolded = input_unfolded.unfold(dim, strel.shape[dim], 1)

    # Sums
    result = input_unfolded - strel

    # Take the maximum
    for dim in range(strel.ndim):
        result, _ = torch.min(result, dim=-1)

    return result


if __name__ == '__main__':
    eroded_image = ErosionFunction.apply(_input_tensor, _strel_tensor, (1, 1), 255)
    eroded_image_old = old_erosion(_input_tensor, _strel_tensor_old, (1, 1), 255)
    eroded_image_python = ErosionPythonFunction.apply(_input_tensor, _strel_tensor_python, (1, 1), 255)

    assert f.l1_loss(eroded_image, eroded_image_old, reduction='sum') == 0
    assert f.l1_loss(eroded_image, eroded_image_python, reduction='sum') == 0

    loss = f.l1_loss(eroded_image, _target_image, reduction='sum')
    loss_old = f.l1_loss(eroded_image_old, _target_image, reduction='sum')
    loss_python = f.l1_loss(eroded_image_python, _target_image, reduction='sum')

    assert loss - loss_old == 0
    assert loss - loss_python == 0

    loss.backward()
    loss_old.backward()
    loss_python.backward()

    print(_strel_tensor.grad)
    print(_strel_tensor_old.grad)
    print(_strel_tensor_python.grad)
    # print(_strel_tensor)
