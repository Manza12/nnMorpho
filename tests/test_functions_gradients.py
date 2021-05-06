import torch
from nnMorpho.operations import erosion, dilation
from nnMorpho.functions import ErosionFunction, DilationFunction

image_size = (5, 8)
strel_size = (3, 3)
origin = (2, 2)

rand = True

image_1 = torch.rand(image_size, device='cuda:0')
image_2 = torch.rand(image_size, device='cuda:0')
image_batch = torch.stack((image_1, image_2), dim=0)

strel_values = torch.rand(strel_size, device='cuda:0')

strel_batch_erosion = strel_values.clone().to('cuda:0')
strel_batch_erosion.requires_grad = True

strel_stack_erosion = strel_values.clone().to('cuda:0')
strel_stack_erosion.requires_grad = True

strel_batch_dilation = strel_values.clone().to('cuda:0')
strel_batch_dilation.requires_grad = True

strel_stack_dilation = strel_values.clone().to('cuda:0')
strel_stack_dilation.requires_grad = True

target_erosion_1 = erosion(image_1, torch.zeros(strel_size, device='cuda:0'))
target_erosion_2 = erosion(image_2, torch.zeros(strel_size, device='cuda:0'))
target_erosion_batch = torch.stack((target_erosion_1, target_erosion_2), dim=0)

target_dilation_1 = dilation(image_1, torch.zeros(strel_size, device='cuda:0'))
target_dilation_2 = dilation(image_2, torch.zeros(strel_size, device='cuda:0'))
target_dilation_batch = torch.stack((target_dilation_1, target_dilation_2), dim=0)


if __name__ == '__main__':
    # Erosion

    eroded_1 = ErosionFunction.apply(image_1, strel_stack_erosion, origin, 1e20)
    eroded_2 = ErosionFunction.apply(image_2, strel_stack_erosion, origin, 1e20)
    eroded_stack = torch.stack((eroded_1, eroded_2), dim=0)

    eroded_batch = ErosionFunction.apply(image_batch, strel_batch_erosion, origin, 1e20)

    criterion = torch.nn.L1Loss(reduction='sum')

    loss_1_erosion = criterion(eroded_1, target_erosion_1)
    loss_2_erosion = criterion(eroded_2, target_erosion_2)
    loss_stack_erosion = loss_1_erosion + loss_2_erosion

    loss_batch_erosion = criterion(eroded_batch, target_erosion_batch)

    loss_stack_erosion.backward()
    loss_batch_erosion.backward()

    print("Gradients for erosion")
    print("Gradient batch:\n", strel_batch_erosion.grad)
    print("Gradient stack:\n", strel_stack_erosion.grad)

    # Dilation

    dilated_1 = DilationFunction.apply(image_1, strel_stack_dilation, origin, 1e20)
    dilated_2 = DilationFunction.apply(image_2, strel_stack_dilation, origin, 1e20)
    dilated_stack = torch.stack((dilated_1, dilated_2), dim=0)

    dilated_batch = DilationFunction.apply(image_batch, strel_batch_dilation, origin, 1e20)

    criterion = torch.nn.L1Loss(reduction='sum')

    loss_1_dilation = criterion(dilated_1, target_dilation_1)
    loss_2_dilation = criterion(dilated_2, target_dilation_2)
    loss_stack_dilation = loss_1_dilation + loss_2_dilation

    loss_batch_dilation = criterion(dilated_batch, target_dilation_batch)

    loss_stack_dilation.backward()
    loss_batch_dilation.backward()

    print("Gradients for dilation")
    print("Gradient batch:\n", strel_batch_dilation.grad)
    print("Gradient stack:\n", strel_stack_dilation.grad)
