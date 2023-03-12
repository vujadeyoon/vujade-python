"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_numpy.py
Description: A module for numpy
"""


import numpy as np


def where_2d(_a: np.ndarray, _b: np.ndarray) -> np.ndarray:
    # https://stackoverflow.com/questions/62625071/find-indices-of-matching-rows-in-2d-array-without-loop-one-shot
    #
    #
    # a = np.array([[2, 1],
    #               [3, 3],
    #               [4, 6],
    #               [4, 8],
    #               [4, 7],
    #               [4, 3]])
    #
    # b = np.array([[4, 6], [4, 7]])
    #
    # res = array([[2],
    #              [4]])
    return np.argwhere(np.isin(_a, _b).all(axis=1))
