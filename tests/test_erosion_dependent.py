from nnMorpho.parameters import *
from nnMorpho.operations import erosion_dependent


def test_erosion_dependent():
    # Test Operations
    print("Testing the erosion dependent of nnMorpho respect to Scipy")

    # Parameters
    input_shape = (512, 1024)
    str_el_shape = (input_shape[0], 5, 5)
    device = 'cuda:0'

    # Start
    input_tensor = torch.rand(input_shape, device=device, dtype=torch.float32)
    str_el = torch.rand(str_el_shape, device=device, dtype=torch.float32)
    origin = None
    border_value = 0.

    # Compute nnMorpho
    result_morpho = erosion_dependent(input_tensor, str_el, origin, border_value)

    # Compute Scipy
    result_scipy = None


if __name__ == '__main__':
    test_erosion_dependent()
