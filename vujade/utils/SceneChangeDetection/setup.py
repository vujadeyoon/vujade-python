"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Nov. 2, 2020.

Title: vujade_utils.py
Version: 0.1.0
Description: set-up for the FFmpeg based scene change detection with the cython

Usage: python3 setup.py build_ext --inplace
"""


import numpy as np
from distutils.core import setup
from Cython.Build import cythonize


if __name__ == '__main__':
    # Usage: python3 setup.py build_ext --inplace

    setup(
        name = 'Cython scene change detection for CPU',
        ext_modules = cythonize('cy_scd.pyx'),
        include_dirs = [np.get_include()]
    )
