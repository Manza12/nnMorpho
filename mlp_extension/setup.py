from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mlp_cuda',
    ext_modules=[
        CUDAExtension(
            name='mlp_cuda',
            sources=[
                'mlp_cuda.cpp',
                'mlp_cuda_kernel.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
