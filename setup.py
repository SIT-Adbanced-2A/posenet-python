from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("centroid", sources=["get_centroid.pyx"], include_dirs=['.', get_include()])
setup(name="centroid", ext_modules=cythonize([ext]))
