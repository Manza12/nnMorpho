from nnMorpho.parameters import *
from nnMorpho.functions import ErosionFunction, DilationFunction
from nnMorpho.operations import fill_border


# Todo: check parameters OK when initializing modules
class Erosion(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=INF):
        super(Erosion, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structuring_element = torch.nn.Parameter(torch.randn(shape))
        self.structuring_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return ErosionFunction.apply(image, self.structuring_element, self.origin, self.border_value)


class Dilation(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Dilation, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structuring_element = torch.nn.Parameter(torch.randn(shape))
        self.structuring_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return DilationFunction.apply(image, self.structuring_element, self.origin, self.border_value)


class Opening(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value='geodesic'):
        super(Opening, self).__init__()
        self.shape = shape
        self.origin = origin
        self.structuring_element = torch.nn.Parameter(torch.randn(shape))
        self.structuring_element.requires_grad = True

        # Fill border value if needed
        border_value_erosion = fill_border(border_value, 'erosion')
        border_value_dilation = fill_border(border_value, 'dilation')
        self.border_value_erosion = border_value_erosion
        self.border_value_dilation = border_value_dilation

    def forward(self, image: Tensor) -> Tensor:
        return DilationFunction.apply(
            ErosionFunction.apply(
                image, self.structuring_element, self.origin, self.border_value_erosion),
            self.structuring_element, self.origin, self.border_value_dilation)


class Closing(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value='geodesic'):
        super(Closing, self).__init__()
        self.shape = shape
        self.origin = origin
        self.structuring_element = torch.nn.Parameter(torch.randn(shape))
        self.structuring_element.requires_grad = True

        # Fill border value if needed
        border_value_erosion = fill_border(border_value, 'erosion')
        border_value_dilation = fill_border(border_value, 'dilation')
        self.border_value_erosion = border_value_erosion
        self.border_value_dilation = border_value_dilation

    def forward(self, image: Tensor) -> Tensor:
        return ErosionFunction.apply(
            DilationFunction.apply(
                image, self.structuring_element, self.origin, self.border_value_dilation),
            self.structuring_element, self.origin, self.border_value_erosion)


def test_learning(image: Union[str, Tensor, NoneType], operation_str: str, structuring_element_form: str,
                  structuring_element_shape: tuple, structuring_element_origin: Union[str, tuple], model_shape: tuple,
                  model_origin: tuple, iterations: int, iterations_per_step: int, plot_start: bool, plot_steps: bool,
                  loss_scale: str, use_border: bool, learning_rate: float, _batched_images: bool):
    """ Test the learning of the modules
        Parameters
        ----------
        :param image: str, Tensor
            The image to test with.
        :param operation_str: str
            The string of the operation to test with. Options are: 'erosion', 'dilation', 'opening' and 'closing'.
        :param structuring_element_form: str
            The form of the structuring element. Options are: 'square', 'cross' and 'rake'
        :param structuring_element_shape: tuple
            The shape os the structuring element. Should be a tuple of two integers.
        :param structuring_element_origin: str, tuple
            The origin of the structuring element. Options are: 'half' for centered origin or a tuple of integers
            marking the origin.
        :param model_shape: tuple
            The shape of the structuring element of the model. Should be bigger than the actual structuring element.
        :param model_origin: tuple
            The origin of the structuring element of the model.
        :param iterations: int
            Number of iterations of the learning process.
        :param iterations_per_step: int
            Number of iterations between prints of the loss.
        :param plot_start: bool
            If True, the structuring element, original image and target image are shown at the beginning.
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
    from nnMorpho.operations import dilation, erosion, opening, closing
    from nnMorpho.utils import plot_image, get_strel, assert_2d_tuple, create_image, assert_positive_integer, \
        log_scale, lin_scale

    # Prints
    print("Learning the structuring element of %r." % operation_str)
    print("Parameters:")
    print("Image size:", image.shape)
    print("Operation:", operation_str)
    print("structuring_element_form:", structuring_element_form)
    print("structuring_element_shape:", structuring_element_shape)
    print("structuring_element_origin:", structuring_element_origin)
    print("model_shape:", model_shape)
    print("model_origin:", model_origin)
    print("iterations:", iterations)
    print("iterations per step:", iterations_per_step)
    print("plot start:", plot_start)
    print("plot steps:", plot_steps)
    print("loss_scale:", loss_scale)
    print("use_border:", use_border)
    print("learning rate:", learning_rate)

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

    # structuring element
    assert_2d_tuple(structuring_element_shape, 'structuring_element_shape', positive=True)
    structuring_element = get_strel(structuring_element_form, structuring_element_shape, increment=5)

    if structuring_element_origin == 'half':
        structuring_element_origin = (structuring_element_shape[0] // 2, structuring_element_shape[1] // 2)
    elif type(structuring_element_origin) == tuple:
        assert_2d_tuple(structuring_element_origin, 'structuring_element_origin')
    else:
        raise ValueError("Invalid parameter structuring_element_origin. Value should be either 'half' either a tuple.")

    if plot_start:
        plot_image(structuring_element, title='structuring element', origin=structuring_element_origin,
                   v_min=-10, v_max=20)

    # Original image
    if not type(image) == NoneType:
        original_image = image.float()

        if plot_start:
            plot_image(original_image, title='Original image')
    else:
        original_image = create_image(plot=False)

        if operation_str in ['erosion', 'opening']:
            original_image = dilation(original_image, structuring_element, structuring_element_origin, -INF)

        if plot_start:
            plot_image(original_image, title='Original image', v_min=0, v_max=255)

    # Target image
    y = operation(original_image, structuring_element, origin=structuring_element_origin, border_value='geodesic')

    if not use_border:
        x_1 = structuring_element_origin[0]
        x_2 = y.shape[0] - structuring_element.shape[0] + structuring_element_origin[0] + 1
        y_1 = structuring_element_origin[1]
        y_2 = y.shape[1] - structuring_element.shape[1] + structuring_element_origin[1] + 1

        target_image = y[x_1:x_2, y_1:y_2]
    else:
        target_image = y
        x_1 = None
        x_2 = None
        y_1 = None
        y_2 = None

    if plot_start:
        plot_image(target_image, title='Target image', v_min=0, v_max=255)

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

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Learning loop
    start = time.time()

    assert_positive_integer(iterations, 'iterations')
    assert_positive_integer(iterations_per_step, 'iterations_per_step')
    assert type(plot_steps) == bool, 'Invalid type of parameter plot_steps; should be boolean.'
    for t in range(iterations):
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
                plot_image(y_predicted, title='Predicted image at iteration %r' % t, show=False, v_min=0, v_max=255)
                plot_image(y, title='Target image', show=False, v_min=0, v_max=255)
                plot_image(model.structuring_element, title='Learned structuring element', v_min=-10, v_max=20)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    if not _batched_images:
        plot_image(original_image, title='Original image', show=False, v_min=0, v_max=255, name='original_image')
        plot_image(target_image, title='Target image', show=False, v_min=0, v_max=255, name='target_image')
        plot_image(output_image, title='Predicted image', show=False,  v_min=0, v_max=255, name='predicted_image')

    plot_image(f.pad(structuring_element,
                     [(model.shape[0] - structuring_element.shape[0]) // 2,
                      (model.shape[0] - structuring_element.shape[0]) // 2,
                      (model.shape[1] - structuring_element.shape[1]) // 2,
                      (model.shape[1] - structuring_element.shape[1]) // 2],
                     mode='constant', value=-INF),
               title='Original structuring element',
               origin=(structuring_element_origin[0] + structuring_element.shape[0] // 2 - 1,
                       structuring_element_origin[1] + structuring_element.shape[1] // 2 - 1),
               show=False, v_min=-10, v_max=20, name='original_structuring_element')

    plot_image(model.structuring_element, title='Learned structuring element', origin=model_origin,
               v_min=-10, v_max=20, name='learned_structuring_element')


if __name__ == '__main__':
    from imageio import imread
    from os.path import join, isfile
    from os import listdir
    from nnMorpho.utils import to_greyscale

    # Operation parameters
    _operations = ['erosion']  # , 'dilation', 'opening', 'closing']

    # structuring element parameters
    _structuring_element_form = 'cross'
    _structuring_element_shape = (5, 5)
    _structuring_element_origin = (_structuring_element_shape[0] // 2, _structuring_element_shape[1] // 2)

    # Model parameters
    _model_shape = (7, 7)
    _model_origin = (_model_shape[0] // 2, _model_shape[1] // 2)

    # Learning parameters
    _iterations = 1000 * 20
    _iterations_per_step = 100
    _plot_start = False
    _plot_steps = False
    _loss_scale = 'lin'
    _learning_rate = 9e-1
    _use_border = True
    _batched_images = True
    _color_images = True

    # Image/s
    if not _batched_images:
        _image = imread(join('..', 'images', 'lena.png'))
        _image = to_greyscale(np.array(_image), warn=False)
        _image_tensor = torch.tensor(_image, device=DEVICE)
    else:
        if _color_images:
            _path = join('..', 'images', 'color')
        else:
            _path = join('..', 'images', 'greyscale')

        _images = [im for im in listdir(_path) if isfile(join(_path, im))]

        # Creation of batched images
        print("\nRecovering images...")

        _images_list = list()
        for im in _images:
            _image = imread(join(_path, im))
            _input_array = to_greyscale(np.array(_image), warn=False).astype(np.float32)
            _input_tensor = torch.tensor(_input_array)

            _images_list.append(_input_tensor)

        _image_tensor = torch.stack(_images_list, 0)
        _image_tensor = _image_tensor.to(DEVICE)

    # Learning
    for _operation_str in _operations:
        print("")
        test_learning(_image_tensor, _operation_str, _structuring_element_form, _structuring_element_shape,
                      _structuring_element_origin, _model_shape, _model_origin, _iterations, _iterations_per_step,
                      _plot_start, _plot_steps, _loss_scale, _use_border, _learning_rate, _batched_images)
