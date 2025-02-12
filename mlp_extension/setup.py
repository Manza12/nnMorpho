from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='mlp_cuda',
    ext_modules=[
        CUDAExtension('mlp_cuda', ['mlp_cuda.cpp', 'mlp_cuda_kernel.cu']),
    ],
    cmdclass={'build_ext': BuildExtension}
)
