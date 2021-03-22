from parameters import *


def name_var(var) -> str:
    return f'{var=}'.split('=')[0]


def assert_positive_integer(variable, name):
    assert type(variable) == int, 'Invalid type of parameter %r; should be an integer.' % name
    assert variable > 0, 'Invalid value of %r; should be greater than 0.' % name


def assert_2d_tuple(variable, name, positive=False):
    assert type(variable) == tuple, 'Invalid type of parameter %r; should be a tuple.' % name
    assert len(variable) == 2, 'Invalid length of parameter %r; should be 2.' % name
    assert type(variable[0]) == int and type(variable[1]) == int, \
        'Invalid type of %r elements; should be integer.' % name
    if positive:
        assert variable[0] > 0 and variable[1] > 0, 'Invalid value of %r elements; should be greater than 0.' % name


def create_image(image_shape: tuple = (64, 64), plot=True) -> Tensor:
    x = torch.multiply(torch.ones(image_shape, device='cuda:0'), -INF)

    n_points = 25
    range_points = [-100, 300]
    points = list()
    for i in range(n_points):
        point_tensor = np.random.rand(2) * 64
        point = point_tensor.astype(int)
        x[point[0], point[1]] = torch.rand(1) * (range_points[1] - range_points[0]) + range_points[0]
        points.append(point)

    if plot:
        plot_image(x, 'Original image')

    return x


def get_strel(form: str, shape: tuple, **kwargs) -> torch.Tensor:
    assert len(shape) == 2, 'Length of shape should be 2.'

    if form == 'square':
        return torch.zeros(shape, device=DEVICE)
    elif form == 'cross':
        try:
            increment = kwargs['increment']
        except KeyError:
            logging.debug('Parameter "increment" does not exist: setting increment to 100.')
            increment = 100

        strel = torch.multiply(torch.ones(shape, device=DEVICE), -INF)

        if shape[0] == shape[1]:
            for i in range(shape[0] // 2 + 1):
                j = i
                strel[i: -i, :] = i * increment
                strel[:, j: -j] = j * increment
        if shape[0] < shape[1]:
            warnings.warn("Not checked option.")
            for i in range(shape[0] // 2 + 1):
                strel[i: -i, :] = i * increment
            for j in range(shape[1] // 2 + 1):
                strel[:, j: -j] = j * increment

            if shape[0] % 2 == 1:
                strel[shape[0] // 2 + 1, :] = (shape[0] // 2 + 1) * increment
            if shape[1] % 2 == 1:
                strel[:, shape[1] // 2 + 1] = (shape[1] // 2 + 1) * increment
        elif shape[0] > shape[1]:
            warnings.warn("Not checked option.")
            for i in range(shape[0] // 2 + 1):
                strel[i: -i, :] = i * increment
            for j in range(shape[1] // 2 + 1):
                strel[:, j: -j] = j * increment

            if shape[1] % 2 == 1:
                strel[:, shape[1] // 2 + 1] = (shape[1] // 2 + 1) * increment
            if shape[0] % 2 == 1:
                strel[shape[0] // 2 + 1, :] = (shape[0] // 2 + 1) * increment

        return strel
    elif form == 'rake':
        try:
            increment = kwargs['increment']
        except KeyError:
            logging.debug('Parameter "increment" does not exist: setting increment to 100.')
            increment = 100

        strel = torch.multiply(torch.ones(shape, device=DEVICE), -INF)
        strel[:, 0] = increment

        for i in range(shape[0] // 2):
            strel[2 * i, :] = i * increment

        return strel

    else:
        raise ValueError('Invalid parameter form.\nAllowed parameters are: "square", "cross" and "rake".')


def plot_image(tensor: torch.Tensor, title, origin=None, show=True, **kwargs):
    import matplotlib.pyplot as plt
    plt.figure()

    try:
        v_min = kwargs['v_min']
        v_max = kwargs['v_max']
        plt.imshow(tensor.cpu().detach().numpy(), cmap='hot', vmin=v_min, vmax=v_max)
    except KeyError:
        plt.imshow(tensor.cpu().detach().numpy(), cmap='hot')

    plt.title(title)
    if origin:
        plt.scatter(origin[0], origin[1], marker='x', c='r')
    if show:
        plt.show()


def to_greyscale(image: np.ndarray, warn=True):
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        if image.shape[2] == 3:
            return image.mean(3)
        elif image.shape[2] == 4:
            if warn:
                warnings.warn('Discarding transparency when converting to grayscale.')
            return image[:, :, :3].mean(2)
