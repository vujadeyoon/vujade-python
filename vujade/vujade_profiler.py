"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_profiler.py
Description: A module for profiler
"""


import os
import re
import traceback
import functools
import time
import datetime
import statistics
import torch
import numpy as np
from vujade import vujade_resource as rsc_
from vujade import vujade_utils as utils_
from vujade.vujade_debug import encode_color, printd


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


def measure_time(_iter: int = 1, _warmup: int = 0):
    """
    Usage: @prof_.measure_time(_iter=1, _warmup=0)
           def test(_arg):
               pass
    Description: This is a decorator which can be used to measured the elapsed time for a callable function.
    """
    if _iter < 1:
        raise ValueError('The _iter, {} should be greater than 0.'.format(_iter))

    def _measure_time(_func):
        @functools.wraps(_func)
        def _wrapper(*args, **kwargs):
            debug_info = DEBUG()
            debug_info.get_file_line()

            result = None
            time_cumsum = 0.0
            for _ in range(_warmup):
                result = _func(*args, **kwargs)

            for _ in range(_iter):
                time_start = time.time()
                result = _func(*args, **kwargs)
                time_end = time.time()
                time_cumsum += (time_end - time_start)

            info_trace_1 = '[{}: {}]:'.format(debug_info.fileName, debug_info.lineNumber)
            info_trace_2 = 'The function, {} is called.'.format(_func.__name__)
            info_trace_3 = 'Total time for {} times: {:.2e} sec. Avg. time: {:.2e} sec.'.format(_iter, time_cumsum, time_cumsum / _iter)
            print(encode_color('{} {} {}'.format(info_trace_1, info_trace_2, info_trace_3)))
            return result
        return _wrapper
    return _measure_time


class IntegratedProfiler(object):
    def __init__(self, _pid: int = utils_.getpid(), _gpu_id: int = 0) -> None:
        super(IntegratedProfiler, self).__init__()
        self.prof_time = TimeProfiler()
        self.prof_mem_main = MainMemoryProfiler(_pid=_pid)
        self.prof_mem_gpu = GPUMemoryProfiler(_pid=_pid, _gpu_id=_gpu_id)

    def run(self, _is_print: bool = False, _is_pause: bool = False):
        self.prof_time.run(_is_print=_is_print, _is_pause=False)
        self.prof_mem_main.run(_is_print=_is_print, _is_pause=False)
        self.prof_mem_gpu.run(_is_print=_is_print, _is_pause=_is_pause)
        utils_.endl()


class MainMemoryProfiler(rsc_.MainMemory, DEBUG):
    def __init__(self, _pid=utils_.getpid()):
        rsc_.MainMemory.__init__(self, _pid=_pid)
        DEBUG.__init__(self)
        self.proc = utils_.getproc(_pid=_pid)
        self.mem_mb_prev = self.get_mem_main_proc()
        self.mem_mb_curr = 0.0
        self.mem_total = self.get_mem_main_total()
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

        info_mem = 'Main memory: {:8.2f} MiB ({:6.2f} %), Memory variation: [{}] {:8.2f} MiB.'.format(self.mem_mb_prev, self.mem_percent_curr, self.mem_desc.ljust(8), self.mem_variation)
        info_trace = '[{}: {}] '.format(self.fileName, self.lineNumber) + info_mem
        _print(info_trace)

    def _update(self):
        self.mem_mb_curr = self.get_mem_main_proc()
        self.mem_percent_curr = 100 * (self.mem_mb_curr / self.mem_total)

        if self.mem_mb_prev < self.mem_mb_curr:
            self.mem_desc = 'Increase'
        elif self.mem_mb_prev > self.mem_mb_curr:
            self.mem_desc = 'Decrease'
        else:
            self.mem_desc = 'Same'

        self.mem_variation = self.mem_mb_curr - self.mem_mb_prev
        self.mem_mb_prev = self.mem_mb_curr


class GPUMemoryProfiler(rsc_.GPUMemory, DEBUG):
    def __init__(self, _pid=utils_.getpid(), _gpu_id=0):
        rsc_.GPUMemory.__init__(self, _pid=_pid, _gpu_id=_gpu_id)
        DEBUG.__init__(self)
        self.mem_total = self.get_mem_gpu_total()
        self.mem_mb_prev = self.get_mem_gpu_proc()
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

        info_mem = 'GPU  memory: {:8.2f} MiB ({:6.2f} %), Memory variation: [{}] {:8.2f} MiB.'.format(self.mem_mb_prev, self.mem_percent_curr, self.mem_desc.ljust(8), self.mem_variation)
        info_trace = '[{}: {}] '.format(self.fileName, self.lineNumber) + info_mem
        _print(info_trace)

    def _update(self):
        self.mem_mb_curr = self.get_mem_gpu_proc()
        self.mem_percent_curr = 100 * (self.mem_mb_curr / self.mem_total)

        if self.mem_mb_prev < self.mem_mb_curr:
            self.mem_desc = 'Increase'
        elif self.mem_mb_prev > self.mem_mb_curr:
            self.mem_desc = 'Decrease'
        else:
            self.mem_desc = 'Same'

        self.mem_variation = self.mem_mb_curr - self.mem_mb_prev
        self.mem_mb_prev = self.mem_mb_curr


class TimeProfiler(DEBUG):
    def __init__(self) -> None:
        super(TimeProfiler, self).__init__()
        self.cnt_call = 0
        self.time_start = 0.0
        self.time_prev = 0.0
        self.time_curr = 0.0
        self.elapsed_time_total = 0.0
        self.elapsed_time_prev = 0.0
        self.elapsed_time_curr = 0.0

    def run(self, _is_print: bool = False, _is_pause: bool = False) -> None:
        if self.cnt_call != 0:
            if _is_pause is True:
                _print = input
            else:
                _print = print

            self.get_file_line()
            self._update()

            info_mem = 'Total time: {:.2f} sec., Time: {:.2f} sec.'.format(self.elapsed_time_total, self.elapsed_time_curr)
            info_trace = '[{}: {}] '.format(self.fileName, self.lineNumber) + info_mem
            _print(info_trace)
        else:
            self.time_start = time.time()
            self.time_prev = self.time_start

        self.cnt_call += 1

    def _update(self):
        self.time_curr = time.time()
        self.elapsed_time_total = self._get_elapsed_time_total()
        self.elapsed_time_curr = self._get_elapsed_time()

        self.time_prev = self.time_curr
        self.elapsed_time_prev = self.elapsed_time_curr

    def _get_elapsed_time(self):
        return self.time_curr - self.time_prev

    def _get_elapsed_time_total(self):
        return self.time_curr - self.time_start


class AverageMeterMainMemory(rsc_.MainMemory):
    def __init__(self, _pid=utils_.getpid(), _warmup=0):
        super(AverageMeterMainMemory, self).__init__(_pid=_pid)
        self.warmup = _warmup
        self.cnt_call = 0
        self.mem_list = []
        self.mem_len = 0
        self.mem_sum = 0.0
        self.mem_avg = 0.0
        self.mem_max = 0.0
        self.proc = utils_.getproc(_pid=_pid)

    def start(self):
        self.mem_start = self.get_mem_main_proc()

    def end(self):
        self.mem_end = self.get_mem_main_proc()
        self.cnt_call += 1

        if self.warmup < self.cnt_call:
            self._update()

    def _update(self):
        self.mem_list.append(self.mem_end - self.mem_start)
        self.mem_len = len(self.mem_list)
        self.mem_sum = sum(self.mem_list)
        self.mem_avg = statistics.mean(self.mem_list)
        self.mem_max = max(self.mem_list)


class AverageMeterGPUMemory(rsc_.GPUMemory):
    def __init__(self, _pid=utils_.getpid(), _gpu_id=0, _warmup=0):
        super(AverageMeterGPUMemory, self).__init__(_pid=_pid, _gpu_id=_gpu_id)
        self.warmup = _warmup
        self.cnt_call = 0
        self.mem_list = []
        self.mem_len = 0
        self.mem_sum = 0.0
        self.mem_avg = 0.0
        self.mem_max = 0.0

    def start(self):
        self.mem_start = self.get_mem_gpu_proc()

    def end(self):
        self.mem_end = self.get_mem_gpu_proc()
        self.cnt_call += 1

        if self.warmup < self.cnt_call:
            self._update()

    def _update(self):
        self.mem_list.append(self.mem_end - self.mem_start)
        self.mem_len = len(self.mem_list)
        self.mem_sum = sum(self.mem_list)
        self.mem_avg = statistics.mean(self.mem_list)
        self.mem_max = max(self.mem_list)


class AverageMeterTime(object):
    """
    This class is intended to profile the processing time
    """
    def __init__(self, _time_scale_factor: float = 1.0, _warmup: int = 0) -> None:
        """
        :param int _warmup: A number of times for warming up.
        """
        super(AverageMeterTime, self).__init__()
        self.time_scale_factor = _time_scale_factor
        self.warmup = _warmup
        self.cnt_call = 0
        self.time_len = 0
        self.time_last = 0.0
        self.time_sum = 0.0
        self.time_avg = 0.0
        self.fps_avg = 0.0
        self.eps_val = 1e-9

    def tic(self) -> None:
        self.time_start = time.time()

    def toc(self) -> None:
        self.time_end = time.time()
        self.cnt_call += 1

        if self.warmup < self.cnt_call:
            self._update()

    def _update(self) -> None:
        self.time_len = self.cnt_call - self.warmup
        self.time_last = self.time_scale_factor * (self.time_end - self.time_start)
        self.time_sum += self.time_last
        self.time_avg = self.time_sum / self.time_len
        self.fps_avg = 1 / (self.time_avg + self.eps_val)


class AverageMeterTimePyTorchGPU(object):
    """
    This class is intended to profile the processing time
    """
    def __init__(self, _time_scale_factor: float = 1.0, _warmup: int = 0) -> None:
        """
        :param int _warmup: A number of times for warming up.
        """
        super(AverageMeterTimePyTorchGPU, self).__init__()
        self.time_scale_factor = _time_scale_factor
        self.warmup = _warmup
        self.cnt_call = 0
        self.time_len = 0
        self.time_last = 0.0
        self.time_sum = 0.0
        self.time_avg = 0.0
        self.fps_avg = 0.0
        self.eps_val = 1e-9
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)

    def tic(self) -> None:
        self.starter.record()

    def toc(self) -> None:
        self.ender.record()
        self._synchronize_gpu()
        self.cnt_call += 1

        if self.warmup < self.cnt_call:
            self._update()

    def _update(self) -> None:
        self.time_len = self.cnt_call - self.warmup
        self.time_last = self.time_scale_factor * self.starter.elapsed_time(self.ender) / 1e3
        self.time_sum += self.time_last
        self.time_avg = self.time_sum / self.time_len
        self.fps_avg = 1 / (self.time_avg + self.eps_val)

    def _synchronize_gpu(self) -> None:
        torch.cuda.synchronize()


class AverageMeterValue(object):
    def __init__(self, **kwargs):
        super(AverageMeterValue, self).__init__()
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


class ETA(object):
    def __init__(self, _len_epoch: int, _num_iters: int, _warmup: int = 0) -> None:
        super(ETA, self).__init__()
        self.len_epoch = _len_epoch
        self.num_iters = _num_iters
        self.warmup = _warmup
        self.avgmeter_time_train = AverageMeterTime(_warmup=self.warmup)
        self.avgmeter_time_valid = AverageMeterTime(_warmup=self.warmup)
        self.time_avg = 0.0

    def tic(self, _is_train: bool = True) -> None:
        if _is_train is True:
            self.avgmeter_time_train.tic()
        else:
            self.avgmeter_time_valid.tic()

    def toc(self, _is_train: bool = True) -> None:
        if _is_train is True:
            self.avgmeter_time_train.toc()
        else:
            self.avgmeter_time_valid.tic()

        self.time_avg = self.avgmeter_time_train.time_avg + (self.avgmeter_time_valid.time_avg / self.len_epoch)

    def get(self, _num_iter_curr: int) -> str:
        num_iter_remain = self.num_iters - _num_iter_curr

        return str(datetime.timedelta(seconds=int(self.time_avg * num_iter_remain)))
