from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='Morphology CUDA',
    version='0.0.1',
    ext_modules=[
        CUDAExtension('morphology_cuda', [
            'morphology.cpp',
            'morphology_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
