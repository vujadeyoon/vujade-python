"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Dec. 03, 2020.
Title: setup.py
Version: 0.1.0
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
