from imageio import imread
from os.path import join, isfile
from os import listdir
from nnMorpho.utils import to_greyscale
from nnMorpho.operations import erosion
from nnMorpho.parameters import *
from nnMorpho.functions import ErosionFunction


_path = join('../..', 'images', 'greyscale')
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
_image_eroded_cpu = erosion(_image_tensor.cpu(), _strel_tensor_cpu, _origin, INF)

_image_batched_eroded_gpu = ErosionFunction.apply(_image_batched_tensor, _strel_tensor_batched_gpu, _origin, INF)
_image_batched_eroded_cpu = erosion(_image_batched_tensor.cpu(), _strel_tensor_batched_cpu, _origin, INF)

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
