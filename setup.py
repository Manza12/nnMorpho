from setuptools import setup
from torch.utils import cpp_extension
setup(
    name='nnMorpho',
    packages=['nnMorpho'],
    version='3.0.3',
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
    ext_modules=[
        cpp_extension.CUDAExtension('greyscale_operators_cpp', [
            'C++/greyscale_operators.cu',
        ]),
        cpp_extension.CUDAExtension('binary_operators_cpp', [
            'C++/binary_operators.cu',
        ])
      ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
