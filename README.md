# nnMorpho
A library for general purpose Mathematical Morphology.

nnMorpho implements tools of greyscale mathematical morphology. It uses the [PyTorch](https://pytorch.org/) framework to take advantage of CUDA tensors and perform GPU computations for improved speed.

Currently, the operations implemented are:
- Erosion
- Dilation
- Opening
- Closing

For the moment, only 2D operations are accepted; however, the input tensors may be of the forms:
- (H, W) (greyscale image)
- (C, H, W) (color image)
- (B, H, W) (batched images)
- (B, C, H, W) (batched color images)

In the future, it is intended to be able to accept n-dimensional operations.

It is important to recall that nnMorpho is designed (for the moment) for being used with a GPU; the efficient code is implemented in CUDA. Nonetheless, efficient CPU implementations will be added when possible; it is intended to be implemented with [SIMD](https://en.wikipedia.org/wiki/SIMD) flow. Currently, the CPU implementations are done by means of the high-memory-consuming [torch.Tensor.unfold](pytorch.org/docs/stable/tensors.html) strategy.

Currently nnMorpho uses float32 type for the tensors. In the future, it is intended to accept float64 and float16 types.

## Installation
As usual, nnMorpho can be installed with 
```bash
pip install nnMorpho
```

However, you should install independently the C++ and CUDA modules that take care of the computations.

To do that, download the C++ and the CUDA folder from the project (you can clone the project for instance), navigate to each these folders from a terminal and run the ```python3 setup.py install``` or ```python setup.py install``` (depending on if you use Unix (Linux/Mac) or Windows operating system). If you use Unix, you may need to edit the script to select your installation folder for Python in the --prefix parameter for the setup.py install. 

### New in version 2.1.1
All the computations are written in C++ or CUDA. Notice that not all the features are curently supported by the two; namely, CUDA supports more features that will be supported in the future by C++.


### New in version 1.1.0
- Operations admit batched color images.

### New in version 1.0.1
- Added partial erosion: this type of erosion is a 2D erosion done with a 1D structural element that varies in the complementary dimension. For instance, if you have a 256x256 image, you set a 5x256 structural element that slides in the first dimension and varies in the second.
- Fixed devices problems raised by [s-rog](https://github.com/s-rog) in the CUDA code (see Issue [#1](https://github.com/Manza12/nnMorpho/issues/1)).

### New in version 1.0.0
This version is first one that uses pure CUDA code to implement the operations. This represents an enormous improvement in the memory usage of the operations: instead of using the [torch.nn.functional.unfold](https://pytorch.org/docs/stable/nn.functional.html#unfold) method or the view-like method [torch.Tensor.unfold](pytorch.org/docs/stable/tensors.html), this version uses pure CUDA code to perform both raw operations and forward and backward implementations for the autograd engine. This allows to both improve the memory usage and the customizability of the operations.

In this version, three modules are the main ones:
- functions.py: here, the PyTorch functions are implemented with forward and backward methods for autograd.
- modules.py: here, the PyTorch modules are implemented with the structural element being the parameter to learn.
- operations.py: here, the raw operations are implemented for PyTorch methods.
