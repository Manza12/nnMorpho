import nnMorpho.functions as MF
import torch


DEVICE = 'cuda:0'  # CPU not working for the moment


class MorphModule(torch.nn.Module):
    def __init__(self, sz=3):
        super(MorphModule, self).__init__()
        self.strel_1 = torch.nn.Parameter(torch.rand(sz, sz, device=DEVICE))
        self.strel_2 = torch.nn.Parameter(torch.rand(sz, sz, device=DEVICE))
        self.strel_list = torch.nn.ParameterList([self.strel_1, self.strel_2])

    def forward(self, x):
        for strel in self.strel_list:
            origin = (strel.shape[0] // 2, strel.shape[1] // 2)
            x = MF.ErosionFunction.apply(x, strel, origin, -1e4)
        return x


if __name__ == '__main__':
    module = MorphModule()
    input_tensor = torch.rand((256, 512), device=DEVICE)
    predicted_tensor = module(input_tensor)
    target_tensor = torch.zeros((256, 512), device=DEVICE)
    criterion = torch.nn.L1Loss()
    loss = criterion(target_tensor, predicted_tensor)
    loss.backward()

    print("Grad of the first strel:\n", module.strel_1.grad)
    print("Grad of the second strel:\n", module.strel_2.grad)
