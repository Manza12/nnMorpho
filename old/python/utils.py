from nnMorpho.parameters import *


def pad_tensor(input_tensor, origin, structural_element, border_value):
    pad_list = []
    for dim in range(structural_element.ndim):
        pad_list += [origin[-dim + 1], structural_element.shape[-dim + 1] - origin[-dim + 1] - 1]
    input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)
    return input_pad


def fill_border(border_value, operation):
    if type(border_value) == str:
        if border_value == 'geodesic':
            if operation == 'erosion':
                border_value = INF
            elif operation == 'dilation':
                border_value = -INF
            else:
                raise ValueError("Invalid operation; should be 'erosion' or 'dilation'")
        elif border_value == 'euclidean':
            border_value = -INF
        else:
            ValueError("Currently string options for border value are: 'geodesic' and 'euclidean'")
    elif type(border_value) in [int, float]:
        pass
    else:
        raise ValueError('The type of the border value should be string, int or float.')

    return border_value


def convert_float(input_tensor, warn=True):
    if not input_tensor.dtype == torch.float32:
        if warn:
            warnings.warn('Casting image type (%r) to float32 since nnMorpho only supports float32 tensors.'
                          % input_tensor.dtype)
        input_tensor = input_tensor.float()

    return input_tensor


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
        plot_image(x, title='Original image')

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


def plot_image(tensor: torch.Tensor, **kwargs):
    import matplotlib.pyplot as plt

    try:
        color = kwargs['color']
        if color:
            tensor = torch.clip(tensor, 0, 255) / 256
    except KeyError:
        pass

    try:
        name = kwargs['name']
        fig = plt.figure(num=name)
    except KeyError:
        fig = plt.figure()

    try:
        cmap = kwargs['cmap']
    except KeyError:
        cmap = 'gray'

    try:
        v_min = kwargs['v_min']
        v_max = kwargs['v_max']
        plt.imshow(tensor.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
    except KeyError:
        plt.imshow(tensor.cpu().detach().numpy(), cmap=cmap)

    try:
        title = kwargs['title']
        fig.suptitle(title)
    except KeyError:
        pass

    try:
        show = kwargs['show']
        if show:
            plt.show()
    except KeyError:
        pass

    try:
        origin = kwargs['origin']
        if origin:
            plt.scatter(origin[0], origin[1], marker='x', c='r')
    except KeyError:
        pass


def plot_images_side_by_side(tensor_1: torch.Tensor, tensor_2: torch.Tensor, **kwargs):
    import matplotlib.pyplot as plt

    try:
        name = kwargs['name']
        fig, (ax_1, ax_2) = plt.subplots(1, 2)
        fig.name(name)
    except KeyError:
        fig, (ax_1, ax_2) = plt.subplots(1, 2)

    try:
        cmap = kwargs['cmap']
    except KeyError:
        cmap = 'gray'

    try:
        v_min = kwargs['v_min']
        v_max = kwargs['v_max']
        ax_1.imshow(tensor_1.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax_2.imshow(tensor_2.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
    except KeyError:
        ax_1.imshow(tensor_1.cpu().detach().numpy(), cmap=cmap)
        ax_2.imshow(tensor_2.cpu().detach().numpy(), cmap=cmap)

    try:
        title = kwargs['title']
        fig.suptitle(title)
    except KeyError:
        pass

    try:
        show = kwargs['show']
        if show:
            plt.show()
    except KeyError:
        pass


def plot_four_operations(input_tensor: torch.Tensor, eroded_tensor: torch.Tensor, dilated_tensor: torch.Tensor,
                         opened_tensor: torch.Tensor, closed_tensor: torch.Tensor, **kwargs):
    import matplotlib.pyplot as plt

    try:
        color = kwargs['color']
        if color:
            input_tensor = torch.clip(input_tensor, 0, 255) / 256
            eroded_tensor = torch.clip(eroded_tensor, 0, 255) / 256
            dilated_tensor = torch.clip(dilated_tensor, 0, 255) / 256
            opened_tensor = torch.clip(opened_tensor, 0, 255) / 256
            closed_tensor = torch.clip(closed_tensor, 0, 255) / 256
    except KeyError:
        pass

    try:
        name = kwargs['name']
        fig = plt.figure(name)
    except KeyError:
        fig = plt.figure()

    gs = fig.add_gridspec(3, 2)

    ax_input = fig.add_subplot(gs[0, :])
    ax_input.set_title('Input image')
    ax_input.set_xticks([])
    ax_input.set_yticks([])

    ax_erosion = fig.add_subplot(gs[1, 0])
    ax_erosion.set_title('Eroded image')
    ax_erosion.set_xticks([])
    ax_erosion.set_yticks([])

    ax_dilation = fig.add_subplot(gs[1, 1])
    ax_dilation.set_title('Dilated image')
    ax_dilation.set_xticks([])
    ax_dilation.set_yticks([])

    ax_opening = fig.add_subplot(gs[2, 0])
    ax_opening.set_title('Opened image')
    ax_opening.set_xticks([])
    ax_opening.set_yticks([])

    ax_closing = fig.add_subplot(gs[2, 1])
    ax_closing.set_title('Closed image')
    ax_closing.set_xticks([])
    ax_closing.set_yticks([])

    try:
        cmap = kwargs['cmap']
    except KeyError:
        cmap = 'gray'

    try:
        v_min = kwargs['v_min']
        v_max = kwargs['v_max']

        ax_input.imshow(input_tensor.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax_erosion.imshow(eroded_tensor.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax_dilation.imshow(dilated_tensor.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax_opening.imshow(opened_tensor.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
        ax_closing.imshow(closed_tensor.cpu().detach().numpy(), cmap=cmap, vmin=v_min, vmax=v_max)
    except KeyError:
        ax_input.imshow(input_tensor.cpu().detach().numpy(), cmap=cmap)
        ax_erosion.imshow(eroded_tensor.cpu().detach().numpy(), cmap=cmap)
        ax_dilation.imshow(dilated_tensor.cpu().detach().numpy(), cmap=cmap)
        ax_opening.imshow(opened_tensor.cpu().detach().numpy(), cmap=cmap)
        ax_closing.imshow(closed_tensor.cpu().detach().numpy(), cmap=cmap)

    try:
        title = kwargs['title']
        fig.suptitle(title)
    except KeyError:
        pass

    try:
        show = kwargs['show']
        if show:
            plt.show()
    except KeyError:
        pass


def to_greyscale(image: np.ndarray, warn=True):
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        if image.shape[2] == 3:
            return np.mean(image, 2)
        elif image.shape[2] == 4:
            if warn:
                warnings.warn('Discarding transparency when converting to grayscale.')
            return image[:, :, :3].mean(2)


def lin_scale(value):
    return value


def log_scale(value):
    return np.log10(value)
