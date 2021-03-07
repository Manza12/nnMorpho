import logging
import sys
import numpy as np
import warnings


def to_greyscale(image: np.ndarray):
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        if image.shape[2] == 3:
            return image.mean(3)
        elif image.shape[2] == 4:
            warnings.warn('Discarding transparency when converting to grayscale.')
            return image[:, :, :3].mean(2)


def configure_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout
    )

    return logging
