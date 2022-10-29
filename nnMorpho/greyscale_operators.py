import greyscale_operators_cpp
from nnMorpho.parameters import *


def erosion(input_tensor: torch.Tensor,
            structuring_element: torch.Tensor,
            footprint: torch.Tensor,
            origin: Optional[Union[tuple, List[int]]] = None,
            border_value: Union[int, float, str] = 'euclidean'):
    """ Erosion is one of the basic operations of Mathematical Morphology. This function computes the grayscale
        erosion of an input tensor by a structuring element.

        Parameters
        ----------
        :param input_tensor: torch.Tensor
            The input tensor to erode. It should be a 2D PyTorch tensor.
        :param structuring_element: torch.Tensor
            The structuring element to erode. The structuring element should be a 2D PyTorch tensor.
        :param footprint: torch.Tensor
            The footprint to erode. The footprint should be a PyTorch tensor with the same dimension as the structuring
            element.
        :param origin: None, tuple, List[int]
            The origin of the structuring element. Default to center of the structuring element.
            Negative indexes are allowed.
        :param border_value: int, float, str
            The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
            - 'geodesic': only points within the input are considered when taking the minimum.
            - 'euclidean': extends naturally the image setting minus infinite value to the border.
            Default value is 'euclidean'.

        Outputs
        -------
        :return: torch.Tensor
            The erosion as a PyTorch tensor of the same shape of the original input.
    """
    # Adapt origin
    if origin is None:
        origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)

    # Compute erosion
    if str(input_tensor.device) == 'cpu':
        result = greyscale_operators_cpp.erosion(input_tensor, structuring_element, footprint,
                                                 origin[0], origin[1], border_value[0])
    else:
        raise NotImplementedError("GPU implementation not done yet")

    return result


# def dilation(input_tensor: torch.Tensor,
#              structuring_element: torch.Tensor,
#              origin: Optional[Union[tuple, List[int]]] = None,
#              border_value: Union[int, float, str] = 'geodesic'):
#     """ Dilation is one of the basic operations of Mathematical Morphology. This function computes the grayscale
#         dilation of an input tensor by a structuring element.
#
#         Parameters
#         ----------
#         :param input_tensor: torch.Tensor
#             The input tensor that you want to dilate. It should be a PyTorch tensor of arbitrary dimension. The
#             dimensions that will be dilated are determined by the structuring element.
#         :param structuring_element: torch.Tensor
#             The structuring element to dilate. The structuring element should be a PyTorch tensor of arbitrary
#             dimension.
#             Its shape should coincide with the shape of the last dimensions of the input_tensor.
#         :param origin: None, tuple, List[int]
#             The origin of the structuring element. Default to center of the structuring element.
#             Negative indexes are allowed.
#         :param border_value: int, float, str
#             The value used to pad the image in the border. Two options are allowed when a string is passed in parameter:
#             - 'geodesic': only points within the input are considered when taking the maximum.
#             - 'euclidean': extends naturally the image setting minus infinite value to the border.
#             Default value is 'geodesic'.
#
#         Outputs
#         -------
#         :return: torch.Tensor
#             The dilation as a PyTorch tensor of the same shape than the original input.
#         """
#     # Check parameters
#     check_parameters(input_tensor, structuring_element, origin, border_value)
#
#     # Adapt origin
#     if not origin:
#         origin = (structuring_element.shape[0] // 2, structuring_element.shape[1] // 2)
#
#     # Fill border value if needed
#     border_value = fill_border(border_value, 'dilation')
#
#     # Convert tensor to float if needed
#     input_tensor = convert_float(input_tensor)
#
#     # Compute the dilation
#     if str(input_tensor.device) == 'cpu':
#         # Pad input
#         m, n = structuring_element.shape
#         o_m, o_n = origin
#         input_pad = pad_tensor(input_tensor, (m - (o_m + 1), n - (o_n + 1)), structuring_element, border_value)
#         result = greyscale_morphology_cpp.greyscale_dilation(input_pad, structuring_element)
#     else:
#         if structuring_element.ndim == 2:
#             # Pad input
#             pad_list = [origin[1], structuring_element.shape[1] - origin[1] - 1,
#                         origin[0], structuring_element.shape[0] - origin[0] - 1]
#             input_pad = f.pad(input_tensor, pad_list, mode='constant', value=border_value)
#
#             if input_tensor.ndim - structuring_element.ndim == 0:
#                 result = morphology_cuda.dilation(input_pad, structuring_element, BLOCK_SHAPE)
#             elif input_tensor.ndim - structuring_element.ndim == 1:
#                 result = morphology_cuda.dilation_batched(input_pad, structuring_element, BLOCK_SHAPE)
#             elif input_tensor.ndim - structuring_element.ndim == 2:
#                 batch_channel_dim = input_pad.shape[0] * input_pad.shape[1]
#                 input_height = input_pad.shape[2]
#                 input_width = input_pad.shape[3]
#                 input_view = input_pad.view(batch_channel_dim, input_height, input_width)
#                 result = morphology_cuda.dilation_batched(input_view, structuring_element, BLOCK_SHAPE)
#                 result = result.view(*input_tensor.shape)
#             else:
#                 raise NotImplementedError("Currently, nnMorpho only supports as input:\n"
#                                           "- 2D tensors of the form (H, W)\n"
#                                           "- 3D tensors of the form (B, H, W)"
#                                           "- 4D tensors of the form (B, C, H, W)")
#         else:
#             raise NotImplementedError("Currently nnMorpho only supports 2D erosion.")
#
#     return result
