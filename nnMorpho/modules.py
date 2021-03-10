from parameters import *
from operations import _erosion, _dilation, _opening, _closing


class Erosion(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=INF):
        super(Erosion, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return _erosion(image, self.structural_element, self.origin, self.border_value)


class Dilation(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Dilation, self).__init__()
        # Todo: check parameters
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return _dilation(image, self.structural_element, self.origin, self.border_value)


class Opening(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Opening, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return _dilation(
            _erosion(image, self.structural_element, self.origin, self.border_value),
            self.structural_element, self.origin, self.border_value)


class Closing(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Closing, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return _erosion(
            _dilation(image, self.structural_element, self.origin, self.border_value),
            self.structural_element, self.origin, self.border_value)


if __name__ == '__main__':
    from utils import plot_image, get_strel

    # Operation
    # Currently dilation is the single one that works great:
    #   - Erosion works for flat structural element
    #   - For the opening, the model uses the trick of putting a Dirac delta as structural element
    #   - The closing goes crazy
    operation_str = 'dilation'  # 'dilation', 'erosion', 'opening', 'closing'

    if operation_str == 'erosion':
        operation = _erosion
        model_operation = Erosion
    elif operation_str == 'dilation':
        operation = _dilation
        model_operation = Dilation
    elif operation_str == 'opening':
        operation = _opening
        model_operation = Opening
    elif operation_str == 'closing':
        operation = _closing
        model_operation = Closing
    else:
        raise ValueError('Invalid operation_str.')

    # Structural element
    structural_element_form = 'rake'  # 'square', 'cross', 'rake'
    structural_element_shape = (7, 7)
    structural_element = get_strel(structural_element_form, structural_element_shape)

    _origin = (structural_element_shape[0] // 2, structural_element_shape[1] // 2)

    plot_image(structural_element, 'Structural element', origin=_origin)

    # Original image
    image_shape = (64, 64)
    x = torch.multiply(torch.ones(image_shape, device='cuda:0'), -INF)

    n_points = 25
    range_points = [-100, 300]
    points = list()
    for i in range(n_points):
        point_tensor = np.random.rand(2) * 64
        point = point_tensor.astype(int)
        x[point[0], point[1]] = torch.rand(1) * (range_points[1] - range_points[0]) + range_points[0]
        points.append(point)

    if operation_str == 'erosion' or operation_str == 'opening':
        x = _dilation(x, structural_element, _origin)

    plot_image(x, 'Original image')

    # Target image
    y = operation(x, structural_element, origin=_origin, border_value=-INF)
    # if operation_str == 'erosion' or operation_str == 'opening':
    #     y = f.threshold(y, -INF / 2, -INF, True)
    plot_image(y, 'Target image')

    # Model
    model = model_operation((15, 15), (7, 7)).to('cuda:0')

    # Loss function
    criterion = torch.nn.MSELoss(reduction='mean')

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # Learning loop
    iterations = 20000
    iterations_per_step = 1000
    plot_steps = False
    for t in range(iterations):
        # Forward pass: Compute predicted y by passing x to the model
        y_predicted = model(x)

        # Compute the loss
        loss = criterion(y_predicted, y)

        if t % iterations_per_step == 0:
            print(t, round(loss.item(), 2))
            if plot_steps:
                plot_image(y_predicted, 'Predicted image at iteration %r' % t)
                plot_image(model.structural_element, 'Learned structural element')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_predicted = model(x)
    loss = criterion(y_predicted, y)
    print('Loss:', round(loss.item(), 2))
    plot_image(x, 'Original image', show=False)
    plot_image(y, 'Target image', show=False)
    plot_image(y_predicted, 'Predicted image', show=False)
    plot_image(structural_element, 'Original structural element', show=False)
    plot_image(model.structural_element, 'Learned structural element')
