"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_str.py
Description: A module for string
"""


import argparse
import json
from typing import Union


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


def str2list(_str: str) -> list:
    if isinstance(_str, str):
        return _str.replace(' ', '').replace('[', '').replace(']', '').split(',')
    else:
        raise argparse.ArgumentTypeError('The argument should be string.')


def strtuple2tuple(_str: str) -> tuple:
    return tuple(json.loads(_str))


def strlist2list(_str: str) -> list:
    return json.loads(_str) # ast.literal_eval(_str_list)


def str2bash_parentheses(_str: str) -> str:
    res = _str.replace('(', '\(')
    res = res.replace(')', '\)')

    return res
