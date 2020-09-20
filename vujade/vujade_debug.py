"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Sep. 20, 2020.

Title: vujade_debug.py
Version: 0.1.1
Description: A module for debug
"""


import os
import psutil
import re
import traceback
import torch
import subprocess
from vujade import vujade_utils as utils_


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


class MainMemoryProfiler(DEBUG):
    def __init__(self, _pid=None):
        super().__init__()

        self.proc = utils_.getproc(_pid=_pid)
        self.mem_mb_prev = self.proc.memory_info().rss / (2.0 ** 20) # == self.proc.memory_info()[0] / (2.0 ** 20), MB
        self.mem_mb_curr = 0.0
        self.mem_desc = None
        self.mem_variation = 0.0
        self.mem_percent_curr = 0.0

    def run(self, _is_print=False, _is_pause=False):
        if _is_pause is True:
            _print = input
        else:
            _print = print

        self.get_file_line()
        self._update()

        info_mem = 'Main memory: {:8.2f} MB ({:6.2f} %), Memory variation: [{}] {:8.2f} MB.'.format(self.mem_mb_prev, self.mem_percent_curr, self.mem_desc.ljust(8), self.mem_variation)
        info_trace = '[{}: {}] '.format(self.fileName, self.lineNumber) + info_mem
        _print(info_trace)

    def _update(self):
        self.mem_mb_curr = self.proc.memory_info().rss / (2.0 ** 20) # MB
        self.mem_percent_curr = dict(psutil.virtual_memory()._asdict())['percent']

        if self.mem_mb_prev < self.mem_mb_curr:
            self.mem_desc = 'Increase'
        elif self.mem_mb_prev > self.mem_mb_curr:
            self.mem_desc = 'Decrease'
        else:
            self.mem_desc = 'Same'

        self.mem_variation = self.mem_mb_curr - self.mem_mb_prev # MB
        self.mem_mb_prev = self.mem_mb_curr


class GPUMemoryProfiler(DEBUG):
    def __init__(self, _pid=None):
        super().__init__()

        self.pid = str(utils_.getpid())
        self.mem_total = self._get_mem_total()
        self.mem_mb_prev = self._get_mem()
        self.mem_mb_curr = 0.0
        self.mem_desc = None
        self.mem_variation = 0.0
        self.mem_percent_curr = 100 * (self.mem_mb_prev / self.mem_total)

    def run(self, _is_print=False, _is_pause=False):
        if _is_pause is True:
            _print = input
        else:
            _print = print

        self.get_file_line()
        self._update()

        info_mem = 'GPU  memory: {:8.2f} MB ({:6.2f} %), Memory variation: [{}] {:8.2f} MB.'.format(self.mem_mb_prev, self.mem_percent_curr, self.mem_desc.ljust(8), self.mem_variation)
        info_trace = '[{}: {}] '.format(self.fileName, self.lineNumber) + info_mem
        _print(info_trace)

    def _update(self):
        self.mem_mb_curr = self._get_mem() # Byte
        self.mem_percent_curr = 100 * (self.mem_mb_curr / self.mem_total)

        if self.mem_mb_prev < self.mem_mb_curr:
            self.mem_desc = 'Increase'
        elif self.mem_mb_prev > self.mem_mb_curr:
            self.mem_desc = 'Decrease'
        else:
            self.mem_desc = 'Same'

        self.mem_variation = self.mem_mb_curr - self.mem_mb_prev # MB
        self.mem_mb_prev = self.mem_mb_curr

    def _get_gpustat(self):
        return str(subprocess.run(['gpustat', '-cp'], stdout=subprocess.PIPE).stdout)

    def _get_mem_total(self):
        gpustat = self._get_gpustat()

        return float(gpustat[gpustat.find('/') + 2:gpustat.find('MB') - 1]) # MB

    def _get_mem(self):
        gpustat = self._get_gpustat()
        gpustat_info = gpustat[gpustat.find(self.pid):]
        mem = gpustat_info[gpustat_info.find('(') + 1:gpustat_info.find('M')]

        if mem == '':
            res = 0
        else:
            res = float(gpustat_info[gpustat_info.find('(') + 1:gpustat_info.find('M')])  # MB

        return res


def debug(_print_str='', _var=None, _is_print_full=False, _is_pause=True, _num_ljust=15):
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




