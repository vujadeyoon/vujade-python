"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: scd.pyx
Description: A module for cython based scene change detection
"""


cimport cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def check_dimension(np.ndarray[np.float32_t, ndim=3] _ndarr_1, np.ndarray[np.float32_t, ndim=3] _ndarr_2):
    cdef int ndarr_1_height = _ndarr_1.shape[0]
    cdef int ndarr_1_width = _ndarr_1.shape[1]
    cdef int ndarr_1_channel = _ndarr_1.shape[2]
    cdef int ndarr_2_height = _ndarr_2.shape[0]
    cdef int ndarr_2_width = _ndarr_2.shape[1]
    cdef int ndarr_2_channel = _ndarr_2.shape[2]

    if ((ndarr_1_height != ndarr_2_height) or (ndarr_1_width != ndarr_2_width) or (ndarr_1_channel != ndarr_2_channel)):
        raise ValueError('The given both frames should have equal shape.')


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def mafd(np.ndarray[np.float32_t, ndim=3] _ndarr_1, np.ndarray[np.float32_t, ndim=3] _ndarr_2, np.float32_t _nb_sad):
    cdef np.float32_t sad_val = np.sum(np.abs(_ndarr_1 - _ndarr_2))
    cdef np.float32_t res = 0.0

    if _nb_sad == 0:
        res = 0.0
    else:
        res = sad_val / _nb_sad

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def diff(np.float32_t _val_1, np.float32_t _val_2):
    cdef np.float32_t res = np.abs(_val_1 - _val_2)

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def calculate_scene_change_value(np.float32_t _mafd, np.float32_t _diff, np.float32_t _min, np.float32_t _max):
    cdef np.float32_t min_val = min(_val_1=_mafd, _val_2=_diff) / 100.0
    cdef np.float32_t res = clip(_val=min_val, _min=_min, _max=_max)

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def min(np.float32_t _val_1, np.float32_t _val_2):
    cdef np.float32_t res = 0.0

    if _val_1 <= _val_2:
        res = _val_1
    else:
        res = _val_2

    return res


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def clip(np.float32_t _val, np.float32_t _min, np.float32_t _max):
    cdef np.float32_t res = _val

    if _val <= _min:
        res = _min
    if _max <= _val:
        res = _max

    return res
