from nnMorpho.parameters import *
from nnMorpho.functions import ErosionFunction, DilationFunction
from nnMorpho.operations import fill_border, _erosion


# Todo: check parameters OK when initializing modules
class Erosion(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=INF):
        super(Erosion, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        if not str(image.device) == 'cpu':
            return ErosionFunction.apply(image, self.structural_element, self.origin, self.border_value)
        else:
            return _erosion(image, self.structural_element, self.origin, self.border_value)


class Dilation(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Dilation, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return DilationFunction.apply(image, self.structural_element, self.origin, self.border_value)


class Opening(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value='geodesic'):
        super(Opening, self).__init__()
        self.shape = shape
        self.origin = origin
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

        # Fill border value if needed
        border_value_erosion = fill_border(border_value, 'erosion')
        border_value_dilation = fill_border(border_value, 'dilation')
        self.border_value_erosion = border_value_erosion
        self.border_value_dilation = border_value_dilation

    def forward(self, image: Tensor) -> Tensor:
        return DilationFunction.apply(
            ErosionFunction.apply(
                image, self.structural_element, self.origin, self.border_value_erosion),
            self.structural_element, self.origin, self.border_value_dilation)


class Closing(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value='geodesic'):
        super(Closing, self).__init__()
        self.shape = shape
        self.origin = origin
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

        # Fill border value if needed
        border_value_erosion = fill_border(border_value, 'erosion')
        border_value_dilation = fill_border(border_value, 'dilation')
        self.border_value_erosion = border_value_erosion
        self.border_value_dilation = border_value_dilation

    def forward(self, image: Tensor) -> Tensor:
        return ErosionFunction.apply(
            DilationFunction.apply(
                image, self.structural_element, self.origin, self.border_value_dilation),
            self.structural_element, self.origin, self.border_value_erosion)
