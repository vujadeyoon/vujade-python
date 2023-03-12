"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_math.py
Description: A module for math
"""


import numpy as np


def get_degree_between_vectors(_ndarr_vec_1: np.ndarray, _ndarr_vec_2: np.ndarray) -> float:
    dot_product = np.dot(_ndarr_vec_1, _ndarr_vec_2)

    mag_vec_1 = np.linalg.norm(_ndarr_vec_1)
    mag_vec_2 = np.linalg.norm(_ndarr_vec_2)

    cosine_angle = dot_product / (mag_vec_1 * mag_vec_2)

    angle_radian = np.arccos(cosine_angle)
    angle_degree = np.degrees(angle_radian)

    return float(angle_degree)