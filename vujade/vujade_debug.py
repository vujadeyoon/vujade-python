"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_debug.py
Description: A module for debug
"""


import os
import re
import traceback
import inspect
import torch
import numpy as np
from vujade import vujade_utils as utils_


class DEBUG(object):
    def __init__(self):
        super(DEBUG, self).__init__()
        self.fileName = None
        self.lineNumber = None
        self.reTraceStack = re.compile('File \"(.+?)\", line (\d+?), .+')

    def get_file_line(self):
        for line in traceback.format_stack()[::-1]:
            m = re.match(self.reTraceStack, line.strip())
            if m:
                fileName = m.groups()[0]

                # ignore case
                if fileName == __file__:
                    continue
                self.fileName = os.path.split(fileName)[1]
                self.lineNumber = m.groups()[1]

                return True

        return False


def printf(*_args, **_kwargs) -> str:
    debug_info = DEBUG()
    debug_info.get_file_line()

    info_str = ''
    for _idx, _arg in enumerate(_args):
        info_str += '{} '.format(_arg)
    info_str = info_str.rstrip(' ')

    info_trace = '[{}: {}]: '.format(debug_info.fileName, debug_info.lineNumber) + info_str

    if ('_is_pause' in _kwargs.keys()) and (_kwargs['_is_pause'] is False):
        _print = print
    else:
        _print = input

    if ('_is_print' in _kwargs.keys()) and (_kwargs['_is_print'] is False):
        pass
    else:
        _print(info_trace)

    return info_trace


def pprintf(*_args, **_kwargs) -> str:
    # Usage: pprintf('var1', 'var2')

    called = inspect.currentframe().f_back.f_locals
    called_keys = called.keys()

    info_str = ''
    for _idx, _arg in enumerate(_args):
        if _arg in called_keys:
            info_str += '{}: {}, '.format(_arg, called[_arg])
        else:
            utils_.print_color(_str='The local variable, {} is not defined.'.format(_arg), _color='WARNING')
    info_str = info_str.rstrip(', ')

    info_trace = printf(info_str, **_kwargs)

    return info_trace


def debug(_print_str='', _var=None, _is_pause=True, _is_print_full=False):
    if _is_pause is True:
        _print = input
    else:
        _print = print

    debug_info = DEBUG()
    debug_info.get_file_line()

    if _var is not None:
        if _is_print_full is False:
            info_var = '{}, {}, {}'.format(type(_var), str(_var.shape), _var.dtype)
        else:
            info_var = '{}'.format(_var)
    else:
        info_var = ''

    info_trace = '[{}: {}]: '.format(debug_info.fileName, debug_info.lineNumber) + _print_str + info_var
    _print(info_trace)


def compare_tensor(_var_1, _var_2, _is_print=True, _is_print_full=False, _is_pause=True):
    diff = _var_1.type(torch.FloatTensor) - _var_2.type(torch.FloatTensor)
    abs_diff = abs(diff)

    if _is_print_full is True:
        print(abs_diff)

    if _is_print is True:
        print('abs_diff_min: {:.2e}, abs_diff_max: {:.2e}, abs_diff_mean: {:.2e}'.format(abs_diff.min().item(), abs_diff.max().item(), abs_diff.mean().item()))

        if _is_pause is True:
            input('Press any key to continue...')

    return abs_diff.min().item(), abs_diff.max().item(), abs_diff.mean().item()


def compare_ndarr(_var_1, _var_2, _is_print=True, _is_print_full=False, _is_pause=True):
    diff = _var_1.astype(np.float32) - _var_2.astype(np.float32)
    abs_diff = abs(diff)

    if _is_print_full is True:
        print(abs_diff)

    if _is_print is True:
        print('abs_diff_min: {:.2e}, abs_diff_max: {:.2e}, abs_diff_mean: {:.2e}'.format(abs_diff.min(), abs_diff.max(), abs_diff.mean()))

        if _is_pause is True:
            input('Press any key to continue...')

    return abs_diff.min(), abs_diff.max(), abs_diff.mean()
