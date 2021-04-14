from parameters import *


class ErosionPythonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        try:
            input_tensor = args[0]
        except IndexError:
            raise Exception('Insufficient parameters: first argument should be the input tensor.')

        try:
            strel_tensor = args[1]
        except IndexError:
            raise Exception('Insufficient parameters: second argument should be the structural element.')

        try:
            origin = args[2]
        except IndexError:
            raise Exception('Insufficient parameters: third argument should be the origin.')

        try:
            border_value = args[3]
        except IndexError:
            raise Exception('Insufficient parameters: fourth argument should be the border value.')

        input_pad = f.pad(input_tensor,
                          (origin[1], strel_tensor.shape[1] - origin[1] - 1,
                           origin[0], strel_tensor.shape[0] - origin[0] - 1),
                          mode='constant', value=border_value)
        output_tensor, indexes = morphology_cuda.erosion_forward(input_pad, strel_tensor, BLOCK_SHAPE)

        strel_shape = torch.tensor(strel_tensor.shape, dtype=torch.int16)
        ctx.save_for_backward(indexes, strel_shape)

        return output_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        indexes, strel_shape = ctx.saved_tensors

        # Handmade
        grad_output = grad_output[0]
        # print(grad_output)
        grad_input = torch.zeros(strel_shape.numpy().tolist(), device='cuda:0')
        for x in range(grad_output.shape[0]):
            for y in range(grad_output.shape[1]):
                index_i = indexes[x, y, 0]
                index_j = indexes[x, y, 1]
                grad_input[index_i, index_j] -= grad_output[x, y]
        #

        result = grad_input

        return None, result, None, None


class ErosionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        try:
            input_tensor = args[0]
        except IndexError:
            raise Exception('Insufficient parameters: first argument should be the input tensor.')

        try:
            strel_tensor = args[1]
        except IndexError:
            raise Exception('Insufficient parameters: second argument should be the structural element.')

        try:
            origin = args[2]
        except IndexError:
            raise Exception('Insufficient parameters: third argument should be the origin.')

        try:
            border_value = args[3]
        except IndexError:
            raise Exception('Insufficient parameters: fourth argument should be the border value.')

        input_pad = f.pad(input_tensor,
                          (origin[1], strel_tensor.shape[1] - origin[1] - 1,
                           origin[0], strel_tensor.shape[0] - origin[0] - 1),
                          mode='constant', value=border_value)

        output_tensor, indexes = morphology_cuda.erosion_forward(input_pad, strel_tensor, BLOCK_SHAPE)

        strel_shape = torch.tensor(strel_tensor.shape, dtype=torch.int16)
        ctx.save_for_backward(indexes, strel_shape)

        return output_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        indexes, strel_shape = ctx.saved_tensors

        result = morphology_cuda.erosion_backward(grad_output[0], indexes, strel_shape, BLOCK_SHAPE)

        return None, result, None, None


class DilationFunction(torch.autograd.Function):
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

        output_tensor, indexes = morphology_cuda.dilation_forward(input_pad, strel_tensor, BLOCK_SHAPE)

        strel_shape = torch.tensor(strel_tensor.shape, dtype=torch.int16)
        ctx.save_for_backward(indexes, strel_shape)

        return output_tensor

    @staticmethod
    def backward(ctx, *grad_output):
        indexes, strel_shape = ctx.saved_tensors

        result = morphology_cuda.dilation_backward(grad_output[0], indexes, strel_shape, BLOCK_SHAPE)

        return None, result, None, None
