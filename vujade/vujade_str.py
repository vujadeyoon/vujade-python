"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_str.py
Description: A module for string
"""


def remove_multiple_whitespace(_str: str) -> str:
    return ' '.join(_str.split())


def str2dict_2(_str: str) -> dict:
    res = {}
    for _idx, _line in enumerate(_str.splitlines()):
        key_val = _line.rstrip().split(None, 1)
        if len(key_val) == 2:
            key = key_val[0]
            val = key_val[1]
            res[key] = val

    return res


def str2dict_1(_str: str) -> dict:
    return dict(line.rstrip().split(None, 1) for line in _str.splitlines())
