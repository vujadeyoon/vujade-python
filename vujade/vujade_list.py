"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_list.py
Description: A module for list
"""


import itertools
import math
import random
import numpy as np
from typing import Any
from collections import deque
from vujade.vujade_debug import printd


def remove_element(_list_src: list, _element_to_be_removed: list) -> list:
    return [_ele for _ele in _list_src if _ele not in _element_to_be_removed]


def sorted_set(_list: list) -> list:
    return list(sorted(set(_list), key=_list.index))


def remove_empty_element(_list: list) -> list:
    return [x for x in _list if x]


def shuffle_together(**kwargs) -> zip:
    """
    Example: list_1, list_2 = list_.shuffle_together(_list_1=list_1, _list_2=list_2)
    """
    res = np.asarray(list(kwargs.values()), dtype=object).T.tolist()
    random.shuffle(res)
    return zip(*res)


def shift(_list: list, _idx: int) -> list:
    temp = deque(_list)
    temp.rotate(_idx)
    res = list(temp)

    return res


def is_have_negative_element(_list: list) -> bool:
    return (0 < sum(1 for _e in _list if _e < 0))


def flatten_list(_list: list) -> list:
    return list(itertools.chain(*_list))


def cast_list(_list: list, _type: type = int) -> list:
    return list(map(_type, _list)) # [_type(i) for i in _list]


def check_element_type_list(_list: list, _type: type = int) -> bool:
    return all(isinstance(idx, _type) for idx in _list)


def is_unique_element_in_list(_list: list) -> bool:
    if (len(set(_list)) == len(_list)):
        res = True
    else:
        res = False

    return res


def list_matching_idx(_list_1: list, _list_2: list) -> list:
    temp = set(_list_1)
    return [i for i, val in enumerate(_list_2) if val in temp]


def round(_list: list, _decimals: int = 0) -> list:
    # It is recommend using np.round() instead of the round() because of floating point precision.
    return list(map(lambda x: np.round(x, decimals=_decimals), _list))


def floor(_list: list, _decimals: int = 0) -> list:
    offset = 10 ** _decimals
    return list(map(lambda x: math.floor(offset * x) / offset, _list))


def find(_matched_element: Any, _list: list) -> list:
    return [_ele for _ele in _list if _ele == _matched_element]


def find_indices(_matched_element: Any, _list: list) -> list:
    return [_idx for _idx, _ele in enumerate(_list) if _ele == _matched_element]


def find_indices_in(_matched_element: Any, _list: list) -> list:
    return [_idx for _idx, _ele in enumerate(_list) if _matched_element in _ele]
