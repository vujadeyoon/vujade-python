"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: setup.py
Description: A module for installing the cython based distance module
"""


import numpy as np
from distutils.core import setup
from Cython.Build import cythonize


if __name__ == '__main__':
    # Usage: python3 setup.py build_ext --inplace

    setup(
        name = 'Cython distance for CPU',
        ext_modules = cythonize('distance.pyx'),
        include_dirs = [np.get_include()]
    )
