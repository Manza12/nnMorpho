from parameters import *
from operations import _erosion, _dilation, _opening, _closing
from modules import Erosion, Dilation, Opening, Closing
from utils import plot_image, get_strel, assert_2d_tuple, create_image, assert_positive_integer


def test_learning(image: Union[str, Tensor, NoneType], operation_str: str, structural_element_form: str,
                  structural_element_shape: tuple, structural_element_origin: Union[str, tuple], model_shape: tuple,
                  model_origin: tuple, iterations: int, iterations_per_step: int, plot_steps: bool, loss_scale: str):
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
        ToDo: complete the parameters
    """

    # Operation
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
    assert_2d_tuple(structural_element_shape, 'structural_element_shape', positive=True)
    structural_element = get_strel(structural_element_form, structural_element_shape)

    if structural_element_origin == 'half':
        structural_element_origin = (structural_element_shape[0] // 2, structural_element_shape[1] // 2)
    elif type(structural_element_origin) == tuple:
        assert_2d_tuple(structural_element_origin, 'structural_element_origin')
    else:
        raise ValueError("Invalid parameter structural_element_origin. Value should be either 'half' either a tuple.")

    plot_image(structural_element, 'Structural element', origin=structural_element_origin)

    # Original image
    if image:
        x = image
    else:
        x = create_image(plot=False)

        if operation_str in ['erosion', 'opening']:
            x = _dilation(x, structural_element, structural_element_origin)

        plot_image(x, 'Original image')

    # Target image
    y = operation(x, structural_element, origin=structural_element_origin, border_value=-INF)

    plot_image(y, 'Target image')

    # Model
    assert_2d_tuple(model_shape, 'model_shape', True)
    assert_2d_tuple(model_origin, 'model_origin')
    model = model_operation(model_shape, model_origin).to(DEVICE)

    # Loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    if loss_scale == 'lin':
        scale = lambda value: value
    elif loss_scale == 'log':
        scale = lambda value: np.log10(value)
    else:
        raise ValueError("Invalid loss scale: options are: 'lin' and 'log'.")

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # Learning loop
    start = time.time()

    assert_positive_integer(iterations, 'iterations')
    assert_positive_integer(iterations_per_step, 'iterations_per_step')
    assert type(plot_steps) == bool, 'Invalid type of parameter plot_steps; should be boolean.'
    for t in range(iterations):
        # Forward pass: Compute predicted y by passing x to the model
        y_predicted = model(x)

        # Compute the loss
        loss = criterion(y_predicted, y)

        if t % iterations_per_step == 0:
            print("Iteration %r" % t, "Loss %r" % round(scale(loss.item()), 2))
            if plot_steps:
                plot_image(y_predicted, 'Predicted image at iteration %r' % t, show=False)
                plot_image(y, 'Target image', show=False)
                plot_image(model.structural_element, 'Learned structural element')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    y_predicted = model(x)
    loss = criterion(y_predicted, y)
    print('Loss:', round(scale(loss.item()), 2))

    end = time.time()
    elapsed_time = round(end-start, 3)
    print('Time to learn: %r seconds.' % elapsed_time)

    # Plots
    plot_image(x, 'Original image', show=False)
    plot_image(y, 'Target image', show=False)
    plot_image(y_predicted, 'Predicted image', show=False)
    plot_image(structural_element, 'Original structural element', show=False)
    plot_image(model.structural_element, 'Learned structural element')


if __name__ == '__main__':
    # Operation parameters
    _operation_str = 'erosion'

    # Structural element parameters
    _structural_element_form = 'rake'
    _structural_element_shape = (7, 7)
    _structural_element_origin = (0, 0)

    # Model parameters
    _model_shape = (15, 15)
    _model_origin = (0, 0)

    # Learning parameters
    _iterations = 10000
    _iterations_per_step = 1000
    _plot_steps = False
    _loss_scale = 'log'

    # Learning
    test_learning(None, _operation_str, _structural_element_form, _structural_element_shape, _structural_element_origin,
                  _model_shape, _model_origin, _iterations, _iterations_per_step, _plot_steps, _loss_scale)
