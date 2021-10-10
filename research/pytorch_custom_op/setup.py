from setuptools import setup
from torch.utils import cpp_extension

setup(name='custom_reduction',
      ext_modules=[cpp_extension.CppExtension('reduction', ['custom_reduction.cpp'])],
      license='Apache License v2.0',
      cmdclass={'build_ext': cpp_extension.BuildExtension})
