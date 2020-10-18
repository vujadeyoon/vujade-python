"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Oct. 18, 2020.

Title: vujade_profiler.py
Version: 0.1.0
Description: A module for profiler
"""


import psutil
import subprocess
import time
import statistics
import numpy as np
from vujade import vujade_debug as debug_
from vujade import vujade_utils as utils_


class MainMemoryProfiler(debug_.DEBUG):
    def __init__(self, _pid=utils_.getpid()):
        # Unit: MB
        super().__init__()
        self.proc = utils_.getproc(_pid=_pid)
        self.mem_mb_prev = self._get_mem()
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
        self.mem_mb_curr = self._get_mem()
        self.mem_percent_curr = dict(psutil.virtual_memory()._asdict())['percent']

        if self.mem_mb_prev < self.mem_mb_curr:
            self.mem_desc = 'Increase'
        elif self.mem_mb_prev > self.mem_mb_curr:
            self.mem_desc = 'Decrease'
        else:
            self.mem_desc = 'Same'

        self.mem_variation = self.mem_mb_curr - self.mem_mb_prev
        self.mem_mb_prev = self.mem_mb_curr

    def _get_mem(self):
        return self.proc.memory_info().rss / (2.0 ** 20) # == self.proc.memory_info()[0] / (2.0 ** 20)


class GPUMemoryProfiler(debug_.DEBUG):
    def __init__(self, _pid=str(utils_.getpid())):
        # Unit: MB
        super().__init__()
        self.pid = _pid
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
        self.mem_mb_curr = self._get_mem()
        self.mem_percent_curr = 100 * (self.mem_mb_curr / self.mem_total)

        if self.mem_mb_prev < self.mem_mb_curr:
            self.mem_desc = 'Increase'
        elif self.mem_mb_prev > self.mem_mb_curr:
            self.mem_desc = 'Decrease'
        else:
            self.mem_desc = 'Same'

        self.mem_variation = self.mem_mb_curr - self.mem_mb_prev
        self.mem_mb_prev = self.mem_mb_curr

    def _get_gpustat(self):
        return str(subprocess.run(['gpustat', '-cp'], stdout=subprocess.PIPE).stdout)

    def _get_mem_total(self):
        gpustat = self._get_gpustat()

        return float(gpustat[gpustat.find('/') + 2:gpustat.find('MB') - 1])

    def _get_mem(self):
        gpustat = self._get_gpustat()
        gpustat_info = gpustat[gpustat.find(self.pid):]
        mem = gpustat_info[gpustat_info.find('(') + 1:gpustat_info.find('M')]

        if mem == '':
            res = 0.0
        else:
            res = float(gpustat_info[gpustat_info.find('(') + 1:gpustat_info.find('M')])

        return res


class AverageMeterMainMemory:
    def __init__(self, _pid=utils_.getpid(), _warmup=0):
        # Unit: MB
        self.warmup = _warmup
        self.cnt_call = 0
        self.mem_list = []
        self.mem_len = 0
        self.mem_sum = 0.0
        self.mem_avg = 0.0
        self.mem_max = 0.0
        self.proc = utils_.getproc(_pid=_pid)

    def start(self):
        self.mem_start = self._get_mem()

    def end(self):
        self.mem_end = self._get_mem()
        self.cnt_call += 1

        if self.warmup < self.cnt_call:
            self._update()

    def _update(self):
        self.mem_list.append(self.mem_end - self.mem_start)
        self.mem_len = len(self.mem_list)
        self.mem_sum = sum(self.mem_list)
        self.mem_avg = statistics.mean(self.mem_list)
        self.mem_max = max(self.mem_list)

    def _get_mem(self):
        return self.proc.memory_info().rss / (2.0 ** 20) # == self.proc.memory_info()[0] / (2.0 ** 20)


class AverageMeterGPUMemory:
    def __init__(self, _pid=str(utils_.getpid()), _warmup=0):
        # Unit: MB
        self.pid = _pid
        self.warmup = _warmup
        self.cnt_call = 0
        self.mem_list = []
        self.mem_len = 0
        self.mem_sum = 0.0
        self.mem_avg = 0.0
        self.mem_max = 0.0

    def start(self):
        self.mem_start = self._get_mem()

    def end(self):
        self.mem_end = self._get_mem()
        self.cnt_call += 1

        if self.warmup < self.cnt_call:
            self._update()

    def _update(self):
        self.mem_list.append(self.mem_end - self.mem_start)
        self.mem_len = len(self.mem_list)
        self.mem_sum = sum(self.mem_list)
        self.mem_avg = statistics.mean(self.mem_list)
        self.mem_max = max(self.mem_list)

    def _get_gpustat(self):
        return str(subprocess.run(['gpustat', '-cp'], stdout=subprocess.PIPE).stdout)

    def _get_mem(self):
        gpustat = self._get_gpustat()
        gpustat_info = gpustat[gpustat.find(self.pid):]
        mem = gpustat_info[gpustat_info.find('(') + 1:gpustat_info.find('M')]

        if mem == '':
            res = 0.0
        else:
            res = float(gpustat_info[gpustat_info.find('(') + 1:gpustat_info.find('M')])

        return res


class AverageMeterTime:
    def __init__(self, _warmup=0):
        self.warmup = _warmup
        self.cnt_call = 0
        self.time_list = []
        self.time_len = 0
        self.time_sum = 0.0
        self.time_avg = 0.0
        self.fps_avg = 0.0
        self.eps_val = 1e-9

    def tic(self):
        self.time_start = time.time()

    def toc(self):
        self.time_end = time.time()
        self.cnt_call += 1

        if self.warmup < self.cnt_call:
            self._update()

    def _update(self):
        self.time_list.append(self.time_end - self.time_start)
        self.time_len = len(self.time_list)
        self.time_sum = sum(self.time_list)
        self.time_avg = statistics.mean(self.time_list)
        self.fps_avg = 1 / (statistics.mean(self.time_list) + self.eps_val)


class AverageMeterValue:
    def __init__(self, **kwargs):
        self.cnt_call = 0
        self.keys = list(kwargs.keys())
        self.len = len(self.keys)
        self.ndarr_vals_add = np.zeros(shape=(1, self.len), dtype=np.float32)

    def add(self, **kwargs):
        keys_add = list(kwargs.keys())
        vals_add = list(kwargs.values())

        if self.keys != keys_add:
            raise ValueError('The input values for dict_keys may be incorrect.')

        self._update(_vals_add=vals_add)

    def _update(self, _vals_add):
        for idx, add_val in enumerate(_vals_add):
            self.ndarr_vals_add[0, idx] = add_val

        if self.cnt_call == 0:
            self.ndarr_vals = self.ndarr_vals_add.copy()
        else:
            self.ndarr_vals = np.append(self.ndarr_vals, self.ndarr_vals_add, axis=0)

        self.ndarr_vals_sum = self.ndarr_vals.sum(axis=0)
        self.ndarr_vals_avg = self.ndarr_vals.mean(axis=0)
        self.ndarr_vals_max = self.ndarr_vals.max(axis=0)
        self.ndarr_vals_min = self.ndarr_vals.min(axis=0)

        self.cnt_call += 1
