from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='morpho_cuda',
    ext_modules=[
        CUDAExtension(
            name='morpho_cuda',
            sources=['morpho_cuda.cpp', 'morpho_cuda_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
