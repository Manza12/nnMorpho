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
            name='greyscale_morphology_extension',
            sources=[
                'greyscale_morphology.cpp',
                'greyscale_morphology_cpu.cpp',
                'greyscale_morphology_cuda.cu'
            ],
        ),
    ]
else:
    print("CUDA not found. Compiling CPU-only version.")
    ext_modules = [
        CppExtension(
            name='greyscale_morphology_extension',
            sources=[
                'greyscale_morphology.cpp',
                'greyscale_morphology_cpu.cpp'
            ]
        ),
    ]

setup(
    name='greyscale_morphology_extension',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
