from parameters import *
from operations import _erosion, _dilation


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
        return _dilation(image, self.structural_element, self.origin, self.border_value)


class Opening(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Opening, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return _dilation(
            _erosion(image, self.structural_element, self.origin, self.border_value),
            self.structural_element, self.origin, self.border_value)


class Closing(Module):
    def __init__(self, shape: tuple, origin: tuple, border_value=-INF):
        super(Closing, self).__init__()
        self.shape = shape
        self.origin = origin
        self.border_value = border_value
        self.structural_element = torch.nn.Parameter(torch.randn(shape))
        self.structural_element.requires_grad = True

    def forward(self, image: Tensor) -> Tensor:
        return _erosion(
            _dilation(image, self.structural_element, self.origin, self.border_value),
            self.structural_element, self.origin, self.border_value)


if __name__ == '__main__':
    pass
