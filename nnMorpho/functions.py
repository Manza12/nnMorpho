from parameters import *


class ErosionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        input_tensor = args[0]
        strel_tensor = args[1]
        origin = args[2]
        border_value = args[3]

        input_pad = f.pad(input_tensor,
                          (origin[1], strel_tensor.shape[1] - origin[1] - 1,
                           origin[0], strel_tensor.shape[0] - origin[0] - 1),
                          mode='constant', value=border_value)

        output_tensor, indexes = morpho_cuda.erosion_forward(input_pad, strel_tensor, BLOCK_SHAPE)

        strel_shape = torch.tensor(strel_tensor.shape, dtype=torch.int16)
        ctx.save_for_backward(indexes, strel_shape)

        return output_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        indexes, strel_shape = ctx.saved_tensors

        result = morpho_cuda.erosion_backward(grad_output, indexes, strel_shape, BLOCK_SHAPE_INT)

        return None, result, None, None
