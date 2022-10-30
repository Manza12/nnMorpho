import torch.cuda

from nnMorpho.parameters import *
from nnMorpho.operations import erosion_dependent, erosion, \
    dilation_dependent, dilation

from scipy.ndimage.morphology import grey_erosion, grey_dilation


# Parameters
input_shape = (256, 512)
str_el_shape = (input_shape[0], 5, 5)
device = 'cuda:0'


def test_erosion_dependent():
    # Test Operations
    print("Testing the erosion dependent of nnMorpho respect to Scipy")

    # Start
    input_tensor = torch.rand(input_shape, device=device, dtype=torch.float32)
    input_array = input_tensor.cpu().numpy()

    str_el_tensor = torch.rand(str_el_shape, device=device, dtype=torch.float32)
    str_el_array = str_el_tensor.cpu().numpy()

    origin = None
    border_value = 0.

    # Compute nnMorpho
    start = time.time()
    result_morpho = erosion_dependent(input_tensor, str_el_tensor, origin, border_value)
    torch.cuda.synchronize()
    time_morpho = time.time() - start
    print('Time to erosion dependent - nnMorpho - : %.3f seconds' % time_morpho)

    # Compute nnMorpho (old)
    start = time.time()
    result_morpho_old = torch.zeros_like(input_tensor)
    for i in range(str_el_tensor.shape[0]):
        result_morpho_old[i, :] = erosion(input_tensor, str_el_tensor[i, :, :], origin, border_value)[i, :]
    torch.cuda.synchronize()
    time_old = time.time() - start
    print('Time to erosion dependent - nnMorpho (old) - : %.3f seconds' % time_old)

    error_old = torch.mean(torch.abs(result_morpho - result_morpho_old))
    print("Error old: %.3f" % error_old)
    print("Acceleration respect to old: x%d" % (time_old / time_morpho))

    # Compute Scipy
    start = time.time()
    result_scipy = np.zeros_like(input_array)
    for i in range(str_el_array.shape[0]):
        result_scipy[i, :] = grey_erosion(input_array, structure=str_el_array[i, :, :], mode='constant')[i, :]
    torch.cuda.synchronize()
    time_scipy = time.time() - start
    print('Time to erosion dependent - Scipy - : %.3f seconds' % time_scipy)
    print("Acceleration respect to Scipy: x%d" % (time_scipy / time_morpho))

    error = np.mean(np.abs(result_morpho.cpu().numpy() - result_scipy))
    print("Error: %.3f" % error)

    check = np.mean(np.abs(result_morpho_old.cpu().numpy() - result_scipy))
    print("Check: %.3f" % check)

    assert error == 0


def test_dilation_dependent():
    # Test Operations
    print("Testing the dilation dependent of nnMorpho respect to Scipy")

    # Start
    input_tensor = torch.rand(input_shape, device=device, dtype=torch.float32)
    input_array = input_tensor.cpu().numpy()

    str_el_tensor = torch.rand(str_el_shape, device=device, dtype=torch.float32)
    str_el_array = str_el_tensor.cpu().numpy()

    origin = None
    border_value = 0.

    # Compute nnMorpho
    start = time.time()
    result_morpho = dilation_dependent(input_tensor, str_el_tensor, origin, border_value)
    torch.cuda.synchronize()
    time_morpho = time.time() - start
    print('Time to dilation dependent - nnMorpho - : %.3f seconds' % time_morpho)

    # Compute nnMorpho (old)
    start = time.time()
    result_morpho_old = torch.zeros_like(input_tensor)
    for i in range(str_el_tensor.shape[0]):
        result_morpho_old[i, :] = dilation(input_tensor, str_el_tensor[i, :, :], origin, border_value)[i, :]
    torch.cuda.synchronize()
    time_old = time.time() - start
    print('Time to dilation dependent - nnMorpho (old) - : %.3f seconds' % time_old)

    error_old = torch.mean(torch.abs(result_morpho - result_morpho_old))
    print("Error old: %.3f" % error_old)
    print("Acceleration respect to old: x%d" % (time_old / time_morpho))

    # Compute Scipy
    start = time.time()
    result_scipy = np.zeros_like(input_array)
    for i in range(str_el_array.shape[0]):
        result_scipy[i, :] = grey_dilation(input_array, structure=str_el_array[i, :, :], mode='constant')[i, :]
    torch.cuda.synchronize()
    time_scipy = time.time() - start
    print('Time to dilation dependent - Scipy - : %.3f seconds' % time_scipy)
    print("Acceleration respect to Scipy: x%d" % (time_scipy / time_morpho))

    error = np.mean(np.abs(result_morpho.cpu().numpy() - result_scipy))
    print("Error: %.3f" % error)

    check = np.mean(np.abs(result_morpho_old.cpu().numpy() - result_scipy))
    print("Check: %.3f" % check)

    assert error == 0


if __name__ == '__main__':
    test_erosion_dependent()
    test_dilation_dependent()
