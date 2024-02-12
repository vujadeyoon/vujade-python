"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: distance.pyx
Description: A module for cython based distance
"""


cimport cython
import numpy as np
cimport numpy as np
import math


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def euclidean(np.ndarray[np.float32_t, ndim=1] _ndarr_1, np.ndarray[np.float32_t, ndim=1] _ndarr_2):
    cdef np.ndarray[np.float32_t, ndim=1] diff = _ndarr_1 - _ndarr_2
    cdef np.float32_t res = np.sqrt(np.dot(diff, diff))

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def euclidean_2d_from_list(list _val_1, list _val_2):
    # cdef float res = math.sqrt(pow((_val_1[0] - _val_2[0]), 2) + pow((_val_1[1] - _val_2[1]), 2))
    cdef float res = pow(pow((_val_1[0] - _val_2[0]), 2) + pow((_val_1[1] - _val_2[1]), 2), 0.5)

    return res