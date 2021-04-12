from parameters import *
from functions import ErosionFunction, DilationFunction


# Todo: check parameters OK when initializing modules
class Erosion(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=INF):
        super(Erosion, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return ErosionFunction.apply(image, self.structural_element, self.origin, self.border_value)


class Dilation(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Dilation, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return DilationFunction.apply(image, self.structural_element, self.origin, self.border_value)


class Opening(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Opening, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return DilationFunction.apply(
            ErosionFunction.apply(
                image, self.structural_element, self.origin, self.border_value),
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
        return ErosionFunction.apply(
            DilationFunction.apply(
                image, self.structural_element, self.origin, self.border_value),
            self.structural_element, self.origin, self.border_value)


def test_learning(image: Union[str, Tensor, NoneType], operation_str: str, structural_element_form: str,
                  structural_element_shape: tuple, structural_element_origin: Union[str, tuple], model_shape: tuple,
                  model_origin: tuple, epochs: int, iterations: int, iterations_per_step: int, plot_steps: bool,
                  plot_epoch: bool, loss_scale: str, use_border: bool, learning_rate: float):
    """ Test the learning of the modules

        Parameters
        ----------
        :param image: str, Tensor
            The image to test with.
        :param operation_str: str
            The string of the operation to test with. Options are: 'erosion', 'dilation', 'opening' and 'closing'.
        :param structural_element_form: str
            The form of the structural element. Options are: 'square', 'cross' and 'rake'
        :param structural_element_shape: tuple
            The shape os the structural element. Should be a tuple of two integers.
        :param structural_element_origin: str, tuple
            The origin of the structural element. Options are: 'half' for centered origin or a tuple of integers marking
            the origin.
        :param model_shape: tuple
            The shape of the structural element of the model. Should be bigger than the actual structural element.
        :param model_origin: tuple
            The origin of the structural element of the model.
        :param epochs: int
            Number of epochs of the learning process.
        :param iterations: int
            Number of iterations of the learning process.
        :param iterations_per_step: int
            Number of iterations between prints of the loss.
        :param plot_steps: bool
            If True, the predicted and target image are shown when printing the loss.
        :param loss_scale: str
            Scale of the loss. Options are 'lin' and 'log'.
        :param use_border: bool
            If False, the border is not used to compute the loss.
        :param learning_rate: float
            The learning rate.

    """
    # Imports
    from operations import dilation, erosion, opening, closing
    from utils import plot_image, get_strel, assert_2d_tuple, create_image, assert_positive_integer, log_scale, \
        lin_scale
    from tqdm.std import tqdm

    # Prints
    print("Learning the structural element of %r." % operation_str)
    print("Parameters:")
    print("structural_element_form:", structural_element_form)
    print("structural_element_shape:", structural_element_shape)
    print("structural_element_origin:", structural_element_origin)
    print("model_shape:", model_shape)
    print("model_origin:", model_origin)
    print("iterations:", iterations)
    print("loss_scale:", loss_scale)
    print("use_border:", use_border)

    # Operation
    if operation_str == 'erosion':
        operation = erosion
        model_operation = Erosion
    elif operation_str == 'dilation':
        operation = dilation
        model_operation = Dilation
    elif operation_str == 'opening':
        operation = opening
        model_operation = Opening
    elif operation_str == 'closing':
        operation = closing
        model_operation = Closing
    else:
        raise ValueError('Invalid operation_str.')

    # Structural element
    assert_2d_tuple(structural_element_shape, 'structural_element_shape', positive=True)
    structural_element = get_strel(structural_element_form, structural_element_shape, increment=5)

    if structural_element_origin == 'half':
        structural_element_origin = (structural_element_shape[0] // 2, structural_element_shape[1] // 2)
    elif type(structural_element_origin) == tuple:
        assert_2d_tuple(structural_element_origin, 'structural_element_origin')
    else:
        raise ValueError("Invalid parameter structural_element_origin. Value should be either 'half' either a tuple.")

    plot_image(structural_element, 'Structural element', origin=structural_element_origin, v_min=-10, v_max=20)

    # Original image
    if not type(image) == NoneType:
        original_image = image.float()
    else:
        original_image = create_image(plot=False)

        if operation_str in ['erosion', 'opening']:
            original_image = dilation(original_image, structural_element, structural_element_origin, -INF)

        plot_image(original_image, 'Original image')

    # Target image
    y = operation(original_image, structural_element, origin=structural_element_origin, border_value=-INF)

    if not use_border:
        x_1 = structural_element_origin[0]
        x_2 = y.shape[0] - structural_element.shape[0] + structural_element_origin[0] + 1
        y_1 = structural_element_origin[1]
        y_2 = y.shape[1] - structural_element.shape[1] + structural_element_origin[1] + 1

        target_image = y[x_1:x_2, y_1:y_2]
    else:
        target_image = y
        x_1 = None
        x_2 = None
        y_1 = None
        y_2 = None

    plot_image(target_image, 'Target image')

    # Model
    assert_2d_tuple(model_shape, 'model_shape', True)
    assert_2d_tuple(model_origin, 'model_origin')
    model = model_operation(model_shape, model_origin).to(DEVICE)

    # Loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    if loss_scale == 'lin':
        scale = lin_scale
    elif loss_scale == 'log':
        scale = log_scale
    else:
        raise ValueError("Invalid loss scale: options are: 'lin' and 'log'.")

    assert_positive_integer(iterations, 'iterations')
    assert_positive_integer(iterations_per_step, 'iterations_per_step')
    assert type(plot_steps) == bool, 'Invalid type of parameter plot_steps; should be boolean.'

    # Learning loop
    start = time.time()
    for e in range(epochs):
        print("\nEpoch %r" % (e+1))

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        for t in tqdm(range(iterations)):
            # Forward pass: Compute predicted y by passing x to the model
            y_predicted = model(original_image)

            # Compute the loss
            if not use_border:
                loss = criterion(y_predicted[x_1:x_2, y_1:y_2], target_image)
            else:
                loss = criterion(y_predicted, target_image)

            if t % iterations_per_step == 0:
                print("Iteration %r" % t, "Loss %r" % round(scale(loss.item()), 2))
                if plot_steps:
                    plot_image(y_predicted, 'Predicted image at iteration %r' % t, show=False)
                    plot_image(y, 'Target image', show=False)
                    plot_image(model.structural_element, 'Learned structural element', v_min=-10, v_max=20)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if plot_epoch:
            y_predicted = model(original_image)
            plot_image(y_predicted, 'Predicted image at the end of the epoch %r' % (e+1), show=False)
            plot_image(y, 'Target image', show=False)
            plot_image(model.structural_element, 'Learned structural element', v_min=-10, v_max=20)

    # Last prediction
    y_predicted = model(original_image)
    if not use_border:
        loss = criterion(y_predicted[x_1:x_2, y_1:y_2], target_image)
        output_image = y_predicted[x_1:x_2, y_1:y_2]
    else:
        loss = criterion(y_predicted, target_image)
        output_image = y_predicted

    print('Loss:', round(scale(loss.item()), 2))

    end = time.time()
    elapsed_time = round(end-start, 3)
    print('Time to learn: %r seconds.' % elapsed_time)

    # Plots
    plot_image(original_image, 'Original image', show=False, v_min=0, v_max=255)
    plot_image(target_image, 'Target image', show=False, v_min=0, v_max=255)
    plot_image(output_image, 'Predicted image', show=False,  v_min=0, v_max=255)
    plot_image(structural_element, 'Original structural element', origin=structural_element_origin, show=False,
               v_min=-10, v_max=20)
    plot_image(model.structural_element, 'Learned structural element', origin=model_origin,
               v_min=-10, v_max=20)


if __name__ == '__main__':
    from imageio import imread
    from os.path import join, isfile
    from os import listdir
    from utils import to_greyscale

    # Images
    folder = join('..', 'images', 'dataset')
    _images = listdir(folder)

    _images_tensor = torch.empty((24, 512, 512), device=DEVICE, dtype=torch.float32)

    i = 0
    for name in _images:
        path = join(folder, name)
        if isfile(path):
            _image = imread(path)
            _image = to_greyscale(np.array(_image), warn=False)
            _images_tensor[i, :, :] = torch.tensor(_image, device=DEVICE, dtype=torch.float32)
            i += 1

    # Operation parameters
    _operation_str = 'dilation'

    # Structural element parameters
    _structural_element_form = 'cross'
    _structural_element_shape = (7, 7)
    _structural_element_origin = (3, 3)

    # Model parameters
    _model_shape = (15, 15)
    _model_origin = (7, 7)

    # Learning parameters
    _epochs = 10
    _iterations = 1000
    _iterations_per_step = 100
    _plot_steps = False
    _plot_epochs = True
    _loss_scale = 'lin'
    _learning_rate = 5e-1
    _use_border = False

    # Learning
    test_learning(_images_tensor, _operation_str, _structural_element_form, _structural_element_shape,
                  _structural_element_origin, _model_shape, _model_origin, _epochs, _iterations, _iterations_per_step,
                  _plot_steps, _plot_epochs, _loss_scale, _use_border, _learning_rate)
