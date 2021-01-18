"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Dec. 03, 2020.
Title: scd.pyx
Version: 0.1.0
Description: A module for cython based scene change detection
"""


cimport cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def mafd(np.ndarray[np.int16_t, ndim=4] _ndarr_1, np.ndarray[np.int16_t, ndim=4] _ndarr_2, np.float32_t _nb_sad):
    cdef np.ndarray[np.int64_t, ndim=1] sad_val = np.sum(np.abs(_ndarr_1 - _ndarr_2), axis=(1, 2, 3))
    cdef np.ndarray[np.float32_t, ndim=1] res = sad_val.astype(np.float32) / _nb_sad

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def diff(np.ndarray[np.float32_t, ndim=1] _mafd_1, np.ndarray[np.float32_t, ndim=1] _mafd_2):
    cdef np.ndarray[np.float32_t, ndim=1] res = np.abs(_mafd_1 - _mafd_2)

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def calculate_scene_change_value(np.ndarray[np.float32_t, ndim=1] _mafd, np.ndarray[np.float32_t, ndim=1] _diff, np.float32_t _min, np.float32_t _max):
    cdef np.ndarray[np.float32_t, ndim=1] res = np.clip(np.minimum(_mafd, _diff) / 100.0, a_min=_min, a_max=_max)

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def get_idx_sc(np.ndarray[np.float32_t, ndim=1] _scene_change_val, np.float32_t _threshold, np.int64_t _offset):
    cdef np.ndarray[np.int64_t, ndim=1] res = np.where(_threshold <= _scene_change_val)[0] + _offset + 2

    return res
