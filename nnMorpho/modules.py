from parameters import *


class Erosion(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=INF):
        super(Erosion, self).__init__()
        self.shape = shape
        self.origin = origin
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

        # Compute the differences
        strel_flatten = torch.flatten(self.structural_element).unsqueeze(0).unsqueeze(-1)
        differences = image_unfolded - strel_flatten

        # Compute the min
        minimum, _ = torch.min(differences, 1)

        # Reshape
        result = torch.reshape(minimum, image.shape)

        return result


class Dilation(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Dilation, self).__init__()
        self.shape = shape
        self.origin = origin
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

        # Compute the max
        maximum, _ = torch.max(sums, 1)

        # Reshape
        result = torch.reshape(maximum, image.shape)

        return result


if __name__ == '__main__':
    from operations import dilation
    from utils import plot_image, get_strel

    # Original image
    image_shape = (64, 64)
    x = torch.multiply(torch.ones(image_shape, device='cuda:0'), -INF)

    n_points = 20
    points = list()
    for i in range(n_points):
        point_tensor = np.random.rand(2) * 64
        point = point_tensor.astype(int)
        x[point[0], point[1]] = 0
        points.append(point)

    plot_image(x, 'Original image')

    # Structural element
    strel_form = 'square'  # 'cross', 'rake'
    strel_shape = (5, 5)
    strel = get_strel(strel_form, strel_shape)

    plot_image(strel, 'Structural element')

    # Target image
    y = dilation(x, strel, origin=(0, 0), border_value=-INF)
    plot_image(y, 'Target image')

    # Model
    model = Dilation((15, 15), (7, 7)).to('cuda:0')

    # Loss function
    criterion = torch.nn.MSELoss(reduction='mean')

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # Learning loop
    iterations = 10000
    iterations_per_step = 1000
    plot_steps = True
    for t in range(iterations):
        # Forward pass: Compute predicted y by passing x to the model
        y_predicted = model(x)

        # Compute the loss
        loss = criterion(y_predicted, y)

        if t % iterations_per_step == 0:
            print(t, round(np.log10(loss.item())))
            if plot_steps:
                plot_image(y_predicted, 'Predicted image at iteration %r' % t)
                plot_image(model.structural_element, 'Learned structural element')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_predicted = model(x)
    plot_image(x, 'Original image', show=False)
    plot_image(y_predicted, 'Predicted image', show=False)
    plot_image(y, 'Target image', show=False)
    plot_image(model.structural_element, 'Learned structural element')
