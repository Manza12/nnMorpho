import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as f
from typing import Union
import numpy as np


EPS = 1e-6
INF = 1e6


class Erosion(Module):
    def __init__(self, shape: tuple, origin: tuple, p: int, border_value=INF):
        super(Erosion, self).__init__()
        self.shape = shape
        self.origin = origin
        self.p = p
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.rand(shape) + EPS)
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        # Pad image
        image_pad = f.pad(image, [self.origin[0], self.shape[0] - self.origin[0] - 1,
                                  self.origin[1], self.shape[1] - self.origin[1] - 1],
                          mode='constant', value=self.border_value)

        # Unfold the image according to the dimension
        image_unfolded = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=self.shape)

        # Compute Lp mean
        strel_flatten = torch.flatten(self.structural_element).unsqueeze(0).unsqueeze(-1)
        differences = image_unfolded - strel_flatten
        differences_p_mean = torch.norm(- torch.exp(differences), p=self.p, dim=1)
        result = torch.reshape(torch.log(differences_p_mean), image.shape)

        return result


class Dilation(Module):
    def __init__(self, shape: tuple, origin: tuple, alpha: Union[int, float], border_value=-INF):
        super(Dilation, self).__init__()
        self.shape = shape
        self.origin = origin
        self.alpha = alpha
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        # Pad image
        image_pad = f.pad(image, [self.origin[0], self.shape[0] - self.origin[0] - 1,
                                  self.origin[1], self.shape[1] - self.origin[1] - 1],
                          mode='constant', value=self.border_value)

        # Unfold the image according to the dimension
        image_unfolded = f.unfold(image_pad.unsqueeze(0).unsqueeze(0), kernel_size=self.shape)

        # Flip structural element
        structural_element = torch.flip(self.structural_element, (0, 1))

        # Compute the sums
        strel_flatten = torch.flatten(structural_element).unsqueeze(0).unsqueeze(-1)
        sums = image_unfolded + strel_flatten

        # Compute the alpha soft-max
        maximum, _ = torch.max(sums, 1)
        # soft_max_alpha = \
        #     torch.sum(sums * torch.exp(self.alpha * sums), dim=1) / torch.sum(torch.exp(self.alpha * sums), dim=1)

        result = torch.reshape(maximum, image.shape)

        return result


def plot_image(tensor, title, show=True):
    plt.figure()
    plt.imshow(tensor.cpu().detach().numpy(), cmap='hot')
    plt.title(title)
    if show:
        plt.show()


if __name__ == '__main__':
    from operations import dilation
    import matplotlib.pyplot as plt

    x = - INF * torch.ones((64, 64), device='cuda:0')

    N = 20
    points = list()
    for i in range(N):
        point_tensor = np.random.rand(2) * 64
        point = point_tensor.astype(int)
        x[point[0], point[1]] = 0
        points.append(point)

    plot_image(x, 'Original image')

    strel = torch.zeros((5, 5), device='cuda:0')

    y = dilation(x, strel, origin=(2, 2), border_value=-INF)
    plot_image(y, 'Target image')

    # Construct our model by instantiating the class defined above
    alpha = 5
    model = Dilation((15, 15), (7, 7), alpha).to('cuda:0')

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    for t in range(20000):
        # Forward pass: Compute predicted y by passing x to the model
        y_predicted = model(x)

        # Compute and print loss
        loss = criterion(y_predicted, y)

        if t % 1000 == 0:
            print(t, round(np.log10(loss.item())))
            # plot_image(y_predicted, 'Predicted image at iteration %r' % t)
            # plot_image(model.structural_element, 'Structural element')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_predicted = model(x)
    plot_image(y_predicted, 'Predicted image', show=False)
    plot_image(y, 'Target image', show=False)
    plot_image(model.structural_element, 'Structural element')
