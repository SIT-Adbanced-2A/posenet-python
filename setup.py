from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("mask", sources=["array_mask_create.pyx"], include_dirs=['.', get_include()])
setup(name="mask", ext_modules=cythonize([ext]))