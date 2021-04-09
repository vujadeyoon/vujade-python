"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_list.py
Description: A module for list
"""


import itertools


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
