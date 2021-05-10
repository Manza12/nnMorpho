import numpy as np
import torch
from imageio import imread
from os.path import join
from nnMorpho.utils import plot_image
from nnMorpho.utils import to_greyscale
from nnMorpho.functions import ErosionFunction


DEVICE = 'cuda:0'
PATH = join('..', 'images', 'tests')
NAME = 'cross_white.png'
FILE_PATH = join(PATH, NAME)
STREL_DIM = (3, 3)
ORIGIN = (STREL_DIM[0] // 2, STREL_DIM[1] // 2)
ITERATIONS = 5 * 1000


class MorphModule(torch.nn.Module):
    def __init__(self, height, width):
        super(MorphModule, self).__init__()
        self.image = torch.nn.Parameter(torch.rand(height, width, device=DEVICE), requires_grad=True)

    def forward(self, structural_element):
        x = ErosionFunction.apply(self.image, structural_element, ORIGIN, 1e20)
        return x


if __name__ == '__main__':
    # Image
    image = imread(FILE_PATH)
    image_greyscale = to_greyscale(np.array(image), warn=False).astype(np.float32)
    image_tensor = torch.tensor(image_greyscale, device=DEVICE)

    # Structural element
    strel = torch.zeros(STREL_DIM, dtype=torch.float32, device=DEVICE)

    # Module
    module = MorphModule(image_tensor.shape[0], image_tensor.shape[1])

    # Loss function
    criterion = torch.nn.MSELoss(reduction='mean')

    # Optimizer
    optimizer = torch.optim.SGD(module.parameters(), lr=5e-1)

    # Loop
    for t in range(ITERATIONS):
        # Forward pass: Compute predicted y by passing x to the model
        image_predicted = module(strel)

        # Compute the loss
        loss = criterion(image_predicted, image_tensor)

        if t % 1000 == 0:
            print("Iteration %r" % t, "Loss %r" % round(loss.item(), 2))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Plot
    plot_image(module.image)
    image_predicted = module(strel)
    plot_image(image_predicted)
