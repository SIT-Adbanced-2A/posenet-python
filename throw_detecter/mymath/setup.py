from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("array_centroid", sources=["array_centroid.pyx"], include_dirs=['.', get_include()])
setup(name="array_centroid", ext_modules=cythonize([ext]))

ext = Extension("mask", sources=["array_mask_create.pyx"], include_dirs=['.', get_include()])
setup(name="mask", ext_modules=cythonize([ext]))