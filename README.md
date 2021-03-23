# nnMorpho
A library for general purpose Mathematical Morphology.

nnMorpho implements tools of greyscale mathematical morphology. It uses the [PyTorch](https://pytorch.org/) framework to take advantage of CUDA tensors and perform GPU computations for improved speed. 

## New in version 0.2.0
This version is caracterized by an enourmous improvement in the memory usage of the operations: instead of using the [torch.nn.functional.unfold](https://pytorch.org/docs/stable/nn.functional.html#unfold) method, we use the view-like method [torch.Tensor.unfold](pytorch.org/docs/stable/tensors.html). This allows us to both improve in memory usage and in customability of the operations, allowing arbitrary dimensional inputs and structural elements.
