from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='morphological_dilation2d',
    ext_modules=[
        CUDAExtension(
            name='morphological_dilation2d',
            sources=[
                'morphological_dilation2d.cpp',
                'morphological_dilation2d_kernel.cu'
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
