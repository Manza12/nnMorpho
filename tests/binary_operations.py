from nnMorpho.parameters import *
from nnMorpho.binary_operations import erosion, dilation
from nnMorpho.cylindric_binary_operations import cylindric_erosion, cylindric_dilation
from scipy.ndimage.morphology import binary_erosion, binary_dilation


def random_tensor(size):
    return (2 * torch.rand(size)).type(torch.IntTensor).type(torch.BoolTensor)


def explicit_strel():
    strel_tensor = torch.zeros((5, 7), dtype=torch.bool)
    strel_tensor[2, 3] = True
    strel_tensor[2, 4] = True
    strel_tensor[1, 3] = True
    strel_tensor[3, 3] = True

    return strel_tensor


def explicit_input():
    input_tensor = torch.zeros((12, 21), dtype=torch.bool)
    input_tensor[11, 1] = True

    # input_tensor[7 + 0, 0 + 0] = True
    # input_tensor[7 + 0, 0 + 1] = True
    # input_tensor[7 - 1, 0 + 0] = True
    # input_tensor[7 + 1, 0 + 0] = True
    #
    # input_tensor[0 + 0, 4 + 0] = True
    # input_tensor[0 + 0, 4 + 1] = True
    # input_tensor[0 - 1, 4 + 0] = True
    # input_tensor[0 + 1, 4 + 0] = True

    return input_tensor


def binary_morphology():
    strel_tensor = random_tensor((3, 5))
    input_tensor = random_tensor((151, 211))

    ground_truth_erosion = binary_erosion(input_tensor.numpy(), strel_tensor.numpy())
    ground_truth_dilation = binary_dilation(input_tensor.numpy(), strel_tensor.numpy())

    output_erosion = erosion(input_tensor, strel_tensor, border_value='euclidean').numpy()
    output_dilation = dilation(input_tensor, strel_tensor, border_value='euclidean').numpy()

    assert (ground_truth_erosion == output_erosion).all()
    assert (ground_truth_dilation == output_dilation).all()


def cylindric_binary_morphology():
    strel_tensor = explicit_strel()
    input_tensor = explicit_input()

    strel_array = strel_tensor.numpy()
    input_array = input_tensor.numpy()

    output_erosion = cylindric_erosion(input_tensor, strel_tensor, border_value='euclidean').numpy()
    output_dilation = cylindric_dilation(input_tensor, strel_tensor, border_value='euclidean').numpy()

    assert np.all(output_dilation == output_dilation)


if __name__ == '__main__':
    binary_morphology()
    cylindric_binary_morphology()
