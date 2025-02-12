from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='morphological_dilation2d',
    ext_modules=[
        CppExtension(
            name='morphological_dilation2d',
            sources=['morphological_dilation2d.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
