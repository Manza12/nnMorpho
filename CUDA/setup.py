from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='morphology',
    ext_modules=[
        CUDAExtension('morphology_cuda', [
            'morphology.cpp',
            'morphology_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
