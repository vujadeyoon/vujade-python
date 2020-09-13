"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Sep. 13, 2020.

Title: vujade_debug.py
Version: 0.1
Description: A module for debug
"""


import os
import re
import traceback
import torch

# UTIL : call stack function for log
reTraceStack = re.compile('File \"(.+?)\", line (\d+?), .+')  # [0] filename, [1] line number


def debug(_print_str='', _var=None, _is_print_full=False, _is_pause=True, _num_ljust=15):
    for line in traceback.format_stack()[::-1]:
        m = re.match(reTraceStack, line.strip())
        if m:
            filename = m.groups()[0]

            # ignore case
            if filename == __file__:
                continue
            filename = os.path.split(filename)[1]
            lineNumber = m.groups()[1]

            if _var is not None:
                if _is_print_full is False:
                    info_var = '{}, {}, {}'.format(type(_var), str(_var.shape).ljust(_num_ljust), _var.dtype)
                else:
                    info_var = '{}'.format(_var)
            else:
                info_var = ''

            info_trace = '[{}: {}]: '.format(filename, lineNumber) + _print_str + info_var
            print(info_trace)

            if _is_pause is True:
                input('Press any key to continue...')

            return True

    return False


def compare(_var1, _var2, _is_print=True, _is_print_full=False, _is_pause=True):
    diff = _var1.type(torch.FloatTensor) - _var2.type(torch.FloatTensor)
    abs_diff = abs(diff)

    if _is_print_full is True:
        print(abs_diff)

    if _is_print is True:
        print('abs_diff_min: {:.2e}, abs_diff_max: {:.2e}, abs_diff_mean: {:.2e}'.format(abs_diff.min().item(), abs_diff.max().item(), abs_diff.mean().item()))

        if _is_pause is True:
            input('Press any key to continue...')

    return abs_diff.min().item(), abs_diff.max().item(), abs_diff.mean().item()
