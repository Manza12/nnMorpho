from parameters import *


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
        for i in range(shape[0] // 2):
            strel[i: -i, :] = i * increment
        for j in range(shape[1] // 2):
            strel[:, j: -j] = j * increment

        if shape[0] <= shape[1]:
            if shape[0] % 2 == 1:
                strel[shape[0] // 2 + 1, :] = (shape[0] // 2 + 1) * increment
            if shape[1] % 2 == 1:
                strel[:, shape[1] // 2 + 1] = (shape[1] // 2 + 1) * increment
        else:
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


def plot_image(tensor: torch.Tensor, title, origin=None, show=True):
    import matplotlib.pyplot as plt
    plt.figure()
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
