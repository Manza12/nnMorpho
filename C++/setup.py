from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='morphology_cpp',
      ext_modules=[
		cpp_extension.CppExtension('binary_morphology_cpp', ['binary_morphology.cpp']),
		cpp_extension.CppExtension('cylindric_binary_morphology_cpp', ['cylindric_binary_morphology.cpp'])
		],
      cmdclass={'build_ext': cpp_extension.BuildExtension}
	  )