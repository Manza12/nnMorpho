# nnMorpho
A library for general purpose Mathematical Morphology.

nnMorpho implements tools of greyscale mathematical morphology. It uses the [PyTorch](https://pytorch.org/) framework to take advantage of CUDA tensors and perform GPU computations for improved speed. 

## New in version 0.1.2
- Creation of modules.py where you can train mathematical morphology layers.
- The origin of structural elements is now allowed to be out of the pixels of the structural elelemt. No check is done for the moment.
