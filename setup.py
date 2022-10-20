from setuptools import setup
from torch.utils import cpp_extension

setup(name='morphology_cpp',
      version='0.0.1',
      ext_modules=[
          cpp_extension.CppExtension('binary_morphology_cpp', ['C++/binary_morphology.cpp']),
          cpp_extension.CppExtension('cylindrical_binary_morphology_cpp', ['C++/cylindrical_binary_morphology.cpp']),
          cpp_extension.CppExtension('greyscale_morphology_cpp', ['C++/greyscale_morphology.cpp'])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )

setup(
    name='morphology_cuda',
    version='0.0.1',
    where='./CUDA',
    ext_modules=[
        cpp_extension.CUDAExtension('morphology_cuda', [
            'CUDA/morphology.cpp',
            'CUDA/morphology_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })

setup(
    name='nnMorpho',
    packages=['nnMorpho'],
    version='2.2.1',
    license='MIT',
    description='A library for GPU-accelerated and Machine-Learning adapted'
                ' Mathematical Morphology',
    author='Gonzalo Romero-García',
    author_email='tritery@hotmail.com',
    url='https://github.com/Manza12/nnMorpho',
    download_url='https://github.com/Manza12/nnMorpho/archive/v_0.1.tar.gz',
    keywords=['Mathematical Morphology', 'PyTorch', 'GPU', 'CUDA'],
    install_requires=[
        'numpy',
        'torch'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
