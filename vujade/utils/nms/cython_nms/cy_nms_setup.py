#!/usr/bin/env python


import numpy as np
from distutils.core import setup
from Cython.Build import cythonize


if __name__ == '__main__':
    # Usage: python3 cy_nms_setup.py build_ext --inplace

    setup(
        name = 'Cython NMS for CPU',
        ext_modules = cythonize('cy_nms.pyx'),
        include_dirs = [np.get_include()]
    )
