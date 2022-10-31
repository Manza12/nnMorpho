import torch
from nnMorpho.greyscale_operators import erosion


device = 'cuda:0'

input_tensor = torch.tensor([
    [11, 12, 13, 14],
    [19, 16, 15, 12],
    [12, 11, 10, 17]
], device=device)

str_el_tensor = torch.tensor([
    [1, 2, 3],
    [9, 7, 5]
], device=device)

footprint = torch.tensor([
    [1, 1, 1],
    [1, 0, 1]
], device=device)

output_tensor = erosion(input_tensor, str_el_tensor, footprint, origin=(1, 1), border='g')

assert torch.all(output_tensor == torch.tensor([
    [7, 2, 3, 4],
    [9, 10, 7, 6],
    [6, 3, 2, 1]
]))
