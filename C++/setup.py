from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='Morphology CPU',
      ext_modules=[
          cpp_extension.CppExtension('binary_morphology_cpp', ['binary_morphology.cpp']),
          cpp_extension.CppExtension('cylindrical_binary_morphology_cpp', ['cylindrical_binary_morphology.cpp']),
          cpp_extension.CppExtension('greyscale_morphology_cpp', ['greyscale_morphology.cpp'])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      )
