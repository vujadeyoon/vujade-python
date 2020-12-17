"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Dec. 17, 2020.

Title: vujade_debug.py
Version: 0.2.0
Description: A module for debug
"""


import os
import re
import traceback
import numpy as np
import torch


# UTIL : call stack function for log
reTraceStack = re.compile('File \"(.+?)\", line (\d+?), .+')  # [0] filename, [1] line number


class DEBUG:
    def __init__(self):
        self.fileName = None
        self.lineNumber = None

    def get_file_line(self):
        for line in traceback.format_stack()[::-1]:
            m = re.match(reTraceStack, line.strip())
            if m:
                fileName = m.groups()[0]

                # ignore case
                if fileName == __file__:
                    continue
                self.fileName = os.path.split(fileName)[1]
                self.lineNumber = m.groups()[1]

                return True

        return False


def debug(_print_str='', _var=None, _is_pause=True, _is_print_full=False, _num_ljust=15):
    if _is_pause is True:
        _print = input
    else:
        _print = print

    debug_info = DEBUG()
    debug_info.get_file_line()

    if _var is not None:
        if _is_print_full is False:
            info_var = '{}, {}, {}'.format(type(_var), str(_var.shape).ljust(_num_ljust), _var.dtype)
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




