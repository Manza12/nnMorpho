import numpy as np
from typing import Optional


def binary_erosion(chroma_roll: np.ndarray,
                   structuring_element: np.ndarray,
                   origin: Optional[tuple] = None,
                   border: Optional[str] = None
                   ) -> np.ndarray:
    if border is None:
        border = 'euclidean'

    if border == 'geodesic':
        index_end_avoid = 0
    elif border == 'euclidean':
        index_end_avoid = structuring_element.shape[1] - 1
    else:
        raise ValueError('Type of border %s not understood' % border)

    if not origin:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    result = np.zeros_like(chroma_roll)

    for x in range(chroma_roll.shape[0]):
        for y in range(chroma_roll.shape[1] - index_end_avoid):
            value = 1
            done = False
            for i in range(structuring_element.shape[0]):
                for j in range(structuring_element.shape[1]):
                    x_ok = 0 <= x + i - origin[0] < chroma_roll.shape[0]
                    y_ok = 0 <= y + j - origin[1] < chroma_roll.shape[1]
                    if x_ok and y_ok:
                        chroma_value = chroma_roll[x + i - origin[0], y + j - origin[1]]
                        structure_value = structuring_element[i - origin[0], j - origin[1]]
                        if chroma_value < structure_value:
                            value = 0
                            done = True
                            break
                if done:
                    result[x, y] = value
                    break
            result[x, y] = value

    return result


def binary_dilation(chroma_roll: np.ndarray,
                    structuring_element: np.ndarray,
                    origin: Optional[tuple] = None,
                    border: Optional[str] = None
                    ) -> np.ndarray:
    if border is None:
        border = 'euclidean'

    if border == 'geodesic':
        index_end_avoid = 0
    elif border == 'euclidean':
        index_end_avoid = 0
    else:
        raise ValueError('Type of border %s not understood' % border)

    if not origin:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    result = np.zeros_like(chroma_roll)

    for x in range(chroma_roll.shape[0]):
        for y in range(chroma_roll.shape[1] - index_end_avoid):
            value = 0
            done = False
            for i in range(structuring_element.shape[0]):
                for j in range(structuring_element.shape[1]):
                    if 0 <= y - j - origin[1] < chroma_roll.shape[1]:
                        chroma_value = chroma_roll[x - i - origin[0], y - j - origin[1]]
                        structure_value = structuring_element[i - origin[0], j - origin[1]]
                        if chroma_value == 1 and structure_value == 1:
                            value = 1
                            done = True
                            break
                if done:
                    result[x, y] = value
                    break
            result[x, y] = value

    return result


def binary_hit_or_miss(chroma_roll: np.ndarray,
                       structuring_element_in: np.ndarray,
                       structuring_element_out: np.ndarray,
                       origin_in: Optional[tuple] = None,
                       origin_out: Optional[tuple] = None,
                       border: Optional[str] = None
                       ) -> np.ndarray:
    erosion = binary_erosion(chroma_roll, structuring_element_in, origin_in, border)
    counter_erosion = binary_erosion(1 - chroma_roll, structuring_element_out, origin_out, border)
    hit_or_miss = erosion * counter_erosion
    return hit_or_miss
