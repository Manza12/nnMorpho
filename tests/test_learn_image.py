from os import listdir

import numpy as np
import torch
from imageio import imread
from os.path import join, isfile
from nnMorpho.utils import plot_image
from nnMorpho.utils import to_greyscale
from nnMorpho.functions import ErosionFunction, DilationFunction


DEVICE = 'cuda:0'

STREL_DIM = (1, 1)
ORIGIN = (STREL_DIM[0] // 2, STREL_DIM[1] // 2)
ITERATIONS = 10 * 1000


FUNCTION = 'erosion'  # 'erosion', 'dilation'
BATCHED = True
COLOR = True


class MorphModule(torch.nn.Module):
    def __init__(self, height, width, color=False, batches=0):
        super(MorphModule, self).__init__()
        if color:
            if batches == 0:
                self.image = torch.nn.Parameter(torch.rand(3, height, width, device=DEVICE), requires_grad=True)
            else:
                self.image = torch.nn.Parameter(torch.rand(batches, 3, height, width, device=DEVICE),
                                                requires_grad=True)
        else:
            if batches == 0:
                self.image = torch.nn.Parameter(torch.rand(height, width, device=DEVICE), requires_grad=True)
            else:
                self.image = torch.nn.Parameter(torch.rand(batches, height, width, device=DEVICE), requires_grad=True)

    def forward(self, structural_element):
        if FUNCTION == 'erosion':
            x = ErosionFunction.apply(self.image, structural_element, ORIGIN, 1e20)
        elif FUNCTION == 'dilation':
            x = DilationFunction.apply(self.image, structural_element, ORIGIN, -1e20)
        else:
            raise Exception

        return x


if __name__ == '__main__':
    # Image/s
    if not COLOR:
        if not BATCHED:
            path = join('..', 'images')
            name = 'mona.png'
            file_path = join(path, name)

            image = imread(file_path)
            image_greyscale = to_greyscale(np.array(image), warn=False).astype(np.float32)
            image_tensor = torch.tensor(image_greyscale, device=DEVICE)
        else:
            path = join('..', 'images', 'greyscale')
            images = [im for im in listdir(path) if isfile(join(path, im))]
            images_list = list()
            for im in images:
                image = imread(join(path, im))
                input_array = to_greyscale(np.array(image), warn=False).astype(np.float32)
                input_tensor = torch.tensor(input_array)

                images_list.append(input_tensor)

            images_tensor = torch.stack(images_list, 0)
            image_tensor = images_tensor.to(DEVICE)

    else:
        if not BATCHED:
            path = join('..', 'images', 'color')
            name = '1.png'
            file_path = join(path, name)

            image = imread(file_path)
            image_tensor = torch.tensor(image, device=DEVICE, dtype=torch.float32).transpose(1, 2).transpose(0, 1)
        else:
            path = join('..', 'images', 'color')
            images = [im for im in listdir(path) if isfile(join(path, im))]
            images_list = list()
            for im in images:
                image = imread(join(path, im))
                image_tensor = torch.tensor(image, device=DEVICE, dtype=torch.float32).transpose(1, 2).transpose(0, 1)

                images_list.append(image_tensor)

            images_tensor = torch.stack(images_list, 0)
            image_tensor = images_tensor.to(DEVICE)

    # Structural element
    strel = torch.zeros(STREL_DIM, dtype=torch.float32, device=DEVICE)

    # Module
    if COLOR:
        if BATCHED:
            module = MorphModule(image_tensor.shape[2], image_tensor.shape[3],
                                 color=True, batches=image_tensor.shape[0])
        else:
            module = MorphModule(image_tensor.shape[1], image_tensor.shape[2], color=True)
    else:
        if BATCHED:
            module = MorphModule(image_tensor.shape[1], image_tensor.shape[2], batches=image_tensor.shape[0])
        else:
            module = MorphModule(image_tensor.shape[0], image_tensor.shape[1])

    # Loss function
    criterion = torch.nn.MSELoss(reduction='mean')

    # Optimizer
    optimizer = torch.optim.SGD(module.parameters(), lr=20e1)

    # Loop
    for t in range(ITERATIONS):
        # Forward pass: Compute predicted y by passing x to the model
        image_predicted = module(strel)

        # Compute the loss
        loss = criterion(image_predicted, image_tensor)

        if t % 100 == 0:
            print("Iteration %r" % t, "Loss %r" % round(loss.item(), 2))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Plot
    if not COLOR:
        if not BATCHED:
            plot_image(module.image)
        else:
            for i in range(module.image.shape[0]):
                plot_image(module.image[i])
    else:
        if not BATCHED:
            plot_image(module.image.transpose(0, 1).transpose(1, 2), color=True)
        else:
            for i in range(module.image.shape[0]):
                plot_image(module.image[i].transpose(0, 1).transpose(1, 2), color=True)

    image_predicted = module(strel)
    if not COLOR:
        if not BATCHED:
            plot_image(image_predicted)
        else:
            for i in range(image_predicted.shape[0]):
                plot_image(image_predicted[i])
    else:
        if not BATCHED:
            plot_image(image_predicted.transpose(0, 1).transpose(1, 2), color=True)
        else:
            for i in range(image_predicted.shape[0]):
                plot_image(image_predicted[i].transpose(0, 1).transpose(1, 2), color=True)
