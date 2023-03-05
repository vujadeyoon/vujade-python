"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_str.py
Description: A module for string
"""


import argparse
import ast
import json
from typing import Optional, Tuple, Union


def str2num(_str_num: str, _func_cast_num: type = int) -> Union[int, float]:
    if (_func_cast_num is not int) and (_func_cast_num is not float):
        raise ValueError('The _func_num should be int or float.')
    try:
        res = _func_cast_num(_str_num)
    except ValueError as e:
        raise ValueError("The _str_num, '{}' cannot be converted to number.".format(_str_num))

    return res


def get_alphabets(_columns: Tuple[str, str]) -> tuple:
    return tuple(map(chr, tuple(range(ord(_columns[0]), ord(_columns[1]) + 1))))


def upper(_str: str, _range: Optional[tuple] = None) -> str:
    if _range is None:
        res = _str.upper()
    else:
        idx_start, idx_end = _range
        res = _str[idx_start:idx_end].upper() + _str[idx_end:]

    return res


def rstrip(_str: str, _chars: Optional[str] = None) -> str:
    return _str.rstrip(_chars)


def lstrip(_str: str, _chars: Optional[str] = None) -> str:
    return _str.lstrip(_chars)


def remove_multiple_whitespace(_str: str) -> str:
    return ' '.join(_str.split())


def str2dict_v1(_str: str) -> dict:
    return dict(line.rstrip().split(None, 1) for line in _str.splitlines())


def str2dict_v2(_str: str) -> dict:
    res = {}
    for _idx, _line in enumerate(_str.splitlines()):
        key_val = _line.rstrip().split(None, 1)
        if len(key_val) == 2:
            key = key_val[0]
            val = key_val[1]
            res[key] = val

    return res


def str2dict_v3(_str: str) -> dict:
    return json.loads(_str)


def str2dict_v4(_str: str) -> dict:
    return ast.literal_eval(_str)


def str2bool(_v: Union[str, bool]) -> bool:
    # This function is equivalent to the built-in function, bool(strtobool()), in the distutils.util.
    if isinstance(_v, bool):
       return _v
    if _v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif _v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2tuple(_str: str) -> tuple:
    if isinstance(_str, str):
        return tuple(_str.replace(' ', '').replace('(', '').replace(')', '').split(','))
    else:
        raise argparse.ArgumentTypeError('The argument should be string.')


def str2list_v1(_str: str) -> list:
    if isinstance(_str, str):
        return _str.replace(' ', '').replace('[', '').replace(']', '').split(',')
    else:
        raise argparse.ArgumentTypeError('The argument should be string.')


def str2list_v2(_str: str) -> list:
    return ast.literal_eval(_str)


def list2str(_list_str: list) -> str:
    if isinstance(_list_str, list):
        return ' '.join(map(str, _list_str))
    else:
        raise argparse.ArgumentTypeError('The argument should be string.')


def strtuple2tuple(_str: str) -> tuple:
    return tuple(json.loads(_str))


def strlist2list(_str: str) -> list:
    return json.loads(_str) # ast.literal_eval(_str_list)
