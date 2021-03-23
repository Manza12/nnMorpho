from distutils.core import setup
setup(
  name='nnMorpho',
  packages=['nnMorpho'],
  version='0.1.2',
  license='MIT',
  description='A library for general purpose Mathematical Morphology',
  author='Gonzalo Romero-García',
  author_email='tritery@hotmail.com',
  url='https://github.com/Manza12/nnMorpho',
  download_url='https://github.com/Manza12/nnMorpho/archive/v_0.1.tar.gz',
  keywords=['Mathematical Morphology', 'PyTorch', 'GPU', 'CUDA'],
  install_requires=[
          'numpy',
          'torch',
          'matplotlib',
          'imageio'
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
