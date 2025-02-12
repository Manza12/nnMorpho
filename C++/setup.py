import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available() and os.getenv('CUDA_HOME') is not None

# Choose appropriate extension type
if cuda_available:
    ext_modules = [
        CUDAExtension(
            name='morphological_dilation2d',
            sources=[
                'morphological_dilation2d.cpp',
                'morphological_dilation2d_kernel.cu'
            ],
        ),
    ]
else:
    print("CUDA not found. Compiling CPU-only version.")
    ext_modules = [
        CppExtension(
            name='morphological_dilation2d',
            sources=['morphological_dilation2d.cpp'],
        ),
    ]

setup(
    name='morphological_dilation2d',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
