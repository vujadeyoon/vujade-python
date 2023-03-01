"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_resource.py
Description: A module for resource

Acknowledgement:
    1. This implementation is highly inspired from manikachandna97 and Josh Lee.
    2. url:
        i)   https://www.geeksforgeeks.org/python-how-to-put-limits-on-memory-and-cpu-usage
        ii)  https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call-in-python
        iii) https://stackoverflow.com/questions/41105733/limit-ram-usage-to-python-program
        iv)  https://towardsdatascience.com/why-you-should-wrap-decorators-in-python-5ac3676835f9
"""


import time
import threading
import resource
import signal
import math
import contextlib
import gpustat
import pynvml as N
from functools import wraps
from vujade import vujade_multithread as multithread_
from vujade import vujade_utils as utils_


class GPUStat(object):
    def __init__(self):
        """
        Unit: MiB
        """
        super(GPUStat, self).__init__()
        self.gpu_stats = gpustat.GPUStatCollection.new_query().jsonify()
        self._init()
        self._add_info()

    def _init(self):
        self.hostname = self.gpu_stats['hostname']
        self.date_time = self.gpu_stats['query_time']
        self.gpus = self.gpu_stats['gpus']
        self.num_gpus = len(self.gpu_stats['gpus'])
        self.driver_version = self._get_driver_version()
        self.gpu_id = None
        self.pid = None

    def _add_info(self):
        self.gpu_stats['driver_version'] = self.driver_version

    def get_info_gpu(self, _gpu_id=0):
        self.gpu_id = _gpu_id

        gpu = self._get_gpu()
        index = gpu['index']
        uuid = gpu['uuid']
        name = gpu['name']
        temperature = gpu['temperature.gpu']
        fan_speed = gpu['fan.speed']
        utilization = gpu['utilization.gpu']
        power = gpu['power.draw']
        power_limit = gpu['enforced.power.limit']
        memory_used = gpu['memory.used']
        memory_total = gpu['memory.total']
        gpu_procs = gpu['processes']
        num_gpu_procs = len(gpu_procs)

        return {'index': index,
                'uuid': uuid,
                'name': name,
                'temperature': temperature,
                'fan_speed': fan_speed,
                'utilization': utilization,
                'power': power,
                'power_limit': power_limit,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'gpu_procs': gpu_procs,
                'num_gpu_procs': num_gpu_procs
                }

    def get_info_proc(self, _pid=utils_.getpid(), _gpu_info=None):
        if _gpu_info is None:
            raise ValueError('The argument, _gpu_info, should be assigned.')

        self.pid = _pid
        gpu_procs = _gpu_info['gpu_procs']
        gpu_proc_index = self._get_gpu_procs_index(_gpu_procs=gpu_procs)

        if gpu_proc_index is None:
            gpu_proc = {'username': None, 'command': None, 'gpu_memory_usage': 0, 'pid': self.pid, 'is_proc': False}
        else:
            gpu_proc = gpu_procs[gpu_proc_index]
            gpu_proc['is_proc'] = True

        return gpu_proc

    def pprint(self, _indent=1):
        utils_.pprint(_obj=self.gpu_stats, _indent=_indent)

    def _get_gpu(self):
        if self.num_gpus <= self.gpu_id:
            raise ValueError('The GPU ID is not valid.')
        return self.gpus[self.gpu_id]

    def _get_gpu_procs_index(self, _gpu_procs):
        res = None
        for idx, proc in enumerate(_gpu_procs):
            if self.pid == proc['pid']:
                res = idx
                break

        return res

    def _get_driver_version(self):
        self._nvml_init()

        try:
            driver_version = self._decode(N.nvmlSystemGetDriverVersion())
        except N.NVMLError:
            driver_version = None

        self._nvml_shutdown()

        return driver_version

    def _nvml_init(self):
        N.nvmlInit()

    def _nvml_shutdown(self):
        N.nvmlShutdown()

    def _decode(self, b):
        if isinstance(b, bytes):
            res = b.decode()
        else:
            res = b

        return res


class MainMemory(object):
    def __init__(self, _pid=utils_.getpid()):
        """
        Unit: MiB
        """
        super(MainMemory, self).__init__()
        self.pid = _pid
        self.proc = utils_.getproc(_pid=self.pid)

    def get_mem_main_proc(self):
        return self._get_mem_main_info(_path='/proc/{}/status'.format(self.pid), _tuple_key=('VmRSS:'))

    def get_mem_main_total(self):
        return self._get_mem_main_info(_path='/proc/meminfo', _tuple_key=('MemTotal:'))

    def get_mem_main_free(self):
        return self._get_mem_main_info(_path='/proc/meminfo', _tuple_key=('MemFree:', 'Buffers:', 'Cached:'))

    def get_mem_main_used(self):
        return (self.get_mem_main_total() - self.get_mem_main_free())

    def _get_mem_main_info(self, _path, _tuple_key):
        with open(_path, 'r') as f:
            res = 0.0
            for _f in f:
                sline = _f.split()
                if str(sline[0]) in (_tuple_key):
                    res += int(sline[1])

            res /= 1024

        return res


class GPUMemory(object):
    def __init__(self, _pid=utils_.getpid(), _gpu_id=0):
        """
        Unit: MiB
        """
        super(GPUMemory, self).__init__()
        self.pid = _pid
        self.gpu_id = _gpu_id
        self._renew()

    def get_mem_gpu_proc(self):
        self._renew()
        return float(self.info_proc['gpu_memory_usage'])

    def get_mem_gpu_total(self):
        return float(self.info_gpu['memory_total'])

    def get_mem_gpu_used(self):
        self._renew()
        return float(self.info_gpu['memory_used'])

    def get_mem_gpu_free(self):
        return (self.get_mem_gpu_total() - self.get_mem_gpu_used())

    def _renew(self):
        self.gpu_stat = GPUStat()
        self.info_gpu = self.gpu_stat.get_info_gpu(_gpu_id=self.gpu_id)
        self.info_proc = self.gpu_stat.get_info_proc(_pid=self.pid, _gpu_info=self.info_gpu)


class LimitRunTime(object):
    def __init__(self, _limit_sec: int, _pid: int = utils_.getpid()) -> None:
        """
        Usage:
            Arguments:
                _lmit_sec: only support second.

            class Test(rsc_.LimitRunTime):
                def __init__(self, _limit_sec, _pid=utils_.getpid()):
                    super(Test, self).__init__(_limit_sec, _pid=_pid)
                    self.pid = _pid

                def run(self):
                    cnt = 0
                    while True:
                        print('[{}]: {}'.format(cnt, self.pid))
                        cnt += 1
        """
        super(LimitRunTime, self).__init__()
        self.limit_sec = int(math.floor(_limit_sec))
        self.pid = _pid
        soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        resource.setrlimit(resource.RLIMIT_CPU, (_limit_sec, hard))
        signal.signal(signal.SIGXCPU, self._time_exceeded)

    def _time_exceeded(self, signo, frame):
        utils_.terminate_proc(_pid=self.pid)


class LimitRunTimeFunc(object):
    def __init__(self, _limit_sec: int) -> None:
        """
        Arguments:
            _lmit_sec: only support second.

        Usage:
            def test(_a, _b):
                time_start = time.time()
                while True:
                    print('elapsed time: {:.2f}, {}'.format(time.time() - time_start, _a + _b))

            limit_func = LimitRunTimeFunc(_limit_sec=1.0)
            limit_func.run(test, (10, 20))
            print('is_success: {}'.format(limit_func.is_success))
        """
        super(LimitRunTimeFunc, self).__init__()
        self.limit_sec = int(math.floor(_limit_sec))
        self.is_success = None

    def run(self, _func, args=(), kwargs={}):
        try:
            with self._time_limit():
                _func(*args, **kwargs)
            self.is_success = True
        except self._TimeoutException as e:
            print('The function [{}] will be terminated because the runtime '
                  'is exceeded its limit [{:.2f}] second.'.format(_func, self.limit_sec))
            self.is_success = False

    @contextlib.contextmanager
    def _time_limit(self):
        signal.signal(signal.SIGALRM, self._signal_handler)
        signal.alarm(self.limit_sec)
        try:
            yield
        finally:
            signal.alarm(0)

    def _signal_handler(self, signum, frame):
        raise self._TimeoutException()

    class _TimeoutException(Exception):
        def __init__(self):
            pass


class LimitRunTimeFuncDecorator(object):
    def __init__(self, _limit_sec: int) -> None:
        """
        Arguments:
            _lmit_sec: only support second.

        Usage:
            @LimitRunTimeFuncDecorator(_limit_sec=1.0)
            def test(_a, _b):
                time_start = time.time()
                while True:
                    print('elapsed time: {:.2f}, {}'.format(time.time() - time_start, _a + _b))
            test(_a=10, _b=20)
        """
        super(LimitRunTimeFuncDecorator, self).__init__()
        self.limit_sec = int(math.floor(_limit_sec))

    def __call__(self, _func):
        @wraps(_func)
        def decorator(*args, **kwargs):
            try:
                with self._time_limit():
                    _func(*args, **kwargs)
            except self._TimeoutException as e:
                print('The function [{}] will be terminated because the runtime '
                      'is exceeded its limit [{:.2f}] second.'.format(_func, self.limit_sec))
        return decorator

    @contextlib.contextmanager
    def _time_limit(self):
        signal.signal(signal.SIGALRM, self._signal_handler)
        signal.alarm(self.limit_sec)
        try:
            yield
        finally:
            signal.alarm(0)

    def _signal_handler(self, signum, frame):
        raise self._TimeoutException()

    class _TimeoutException(Exception):
        def __init__(self):
            pass


class LimitMainMemory(MainMemory, multithread_.BaseThread, threading.Thread):
    def __init__(self, _limit_mem, _unit_sec=10e-3, _pid=utils_.getpid(), _gpu_id=0, _is_print=True):
        """
        Unit: MiB

        Usage: LimitMainMemory(_limit_mem=1024).start() # 1024 MiB
        """
        MainMemory.__init__(self, _pid=_pid)
        multithread_.BaseThread.__init__(self)
        threading.Thread.__init__(self)
        self.daemon = True
        self.limit_mem = _limit_mem
        self.unit_sec = _unit_sec
        self.pid = _pid
        self.gpu_id = _gpu_id
        self.is_print = _is_print
        self.mem_curr = 0.0
        self.is_lock = None
        self.is_terminate = None
        self._lock()
        self._unset_terminate()

    def run(self):
        while self.is_lock is True:
            self.mem_curr = self.get_mem_main_proc()
            if self.limit_mem <= self.mem_curr:
                if self.is_print is True:
                    print('The process [{}] will be terminated because the allocated main memory [{:.2f}] MiB '
                          'is exceeded its limit [{:.2f}] MiB.'.format(self.pid, self.mem_curr, self.limit_mem))
                self._unlock()
                self._set_terminate()
            time.sleep(self.unit_sec)

        if self.is_terminate is True:
            utils_.terminate_proc(_pid=self.pid)

    def _lock(self):
        self.is_lock = True

    def _unlock(self):
        self.is_lock = False

    def _set_terminate(self):
        self.is_terminate = True

    def _unset_terminate(self):
        self.is_terminate = False


class LimitMainMemoryDecorator(MainMemory, multithread_.BaseThread, threading.Thread):
    def __init__(self, _limit_mem, _unit_sec=10e-3, _pid=utils_.getpid(), _gpu_id=0, _is_print=True):
        """
        Unit: MiB

        Usage:
            @LimitMainMemoryDecorator(_limit_mem=1024) # 1024 MiB
            def test():
                a = np.zeros([10, 3, 1080, 1920])
                b = np.zeros([10, 3, 1080, 1920])
                c = a + b
                d = 2 * c + b
        """
        MainMemory.__init__(self, _pid=_pid)
        multithread_.BaseThread.__init__(self)
        threading.Thread.__init__(self)
        self.daemon = False
        self.limit_mem = _limit_mem
        self.unit_sec = _unit_sec
        self.pid = _pid
        self.gpu_id = _gpu_id
        self.is_print = _is_print
        self.mem_curr = 0.0
        self.is_lock = None
        self.is_terminate = None
        self._lock()
        self._unset_terminate()
        self.start()

    def __call__(self, _func):
        @wraps(_func)
        def decorator(*args, **kwargs):
            _func(*args, **kwargs)
            self._unlock()
            self.join()
        return decorator

    def run(self):
        while self.is_lock is True:
            self.mem_curr = self.get_mem_main_proc()
            if self.limit_mem <= self.mem_curr:
                if self.is_print is True:
                    print('The process [{}] will be terminated because the allocated main memory [{:.2f}] MiB '
                          'is exceeded its limit [{:.2f}] MiB.'.format(self.pid, self.mem_curr, self.limit_mem))
                self._unlock()
                self._set_terminate()
            time.sleep(self.unit_sec)

        if self.is_terminate is True:
            utils_.terminate_proc(_pid=self.pid)

    def _lock(self):
        self.is_lock = True

    def _unlock(self):
        self.is_lock = False

    def _set_terminate(self):
        self.is_terminate = True

    def _unset_terminate(self):
        self.is_terminate = False


class LimitGPUMemory(GPUMemory, multithread_.BaseThread, threading.Thread):
    def __init__(self, _limit_mem, _unit_sec=10e-3, _pid=utils_.getpid(), _gpu_id=0, _is_print=True):
        """
        Unit: MiB

        Usage: LimitGPUMemory(_limit_mem=1024).start() # 1024 MiB
        """
        GPUMemory.__init__(self, _pid=_pid, _gpu_id=_gpu_id)
        multithread_.BaseThread.__init__(self)
        threading.Thread.__init__(self)
        self.daemon = True
        self.limit_mem = _limit_mem
        self.unit_sec = _unit_sec
        self.pid = _pid
        self.gpu_id = _gpu_id
        self.is_print = _is_print
        self.mem_curr = 0.0
        self.is_lock = None
        self.is_terminate = None
        self._lock()
        self._unset_terminate()

    def run(self):
        while self.is_lock is True:
            self.mem_curr = self.get_mem_gpu_proc()
            if self.limit_mem <= self.mem_curr:
                if self.is_print is True:
                    print('The process [{}] will be terminated because the allocated GPU memory [{:.2f}] MiB '
                          'is exceeded its limit [{:.2f}] MiB.'.format(self.pid, self.mem_curr, self.limit_mem))
                self._unlock()
                self._set_terminate()
            time.sleep(self.unit_sec)

        if self.is_terminate is True:
            utils_.terminate_proc(_pid=self.pid)

    def _lock(self):
        self.is_lock = True

    def _unlock(self):
        self.is_lock = False

    def _set_terminate(self):
        self.is_terminate = True

    def _unset_terminate(self):
        self.is_terminate = False


class LimitGPUMemoryDecorator(GPUMemory, multithread_.BaseThread, threading.Thread):
    def __init__(self, _limit_mem, _unit_sec=10e-3, _pid=utils_.getpid(), _gpu_id=0, _is_print=True):
        """
        Unit: MiB
        Usage:
            @LimitGPUMemoryDecorator(_limit_mem=1024)
            def test():
                a = torch.zeros([10, 3, 1080, 1920], device='cuda:0')
                b = torch.zeros([10, 3, 1080, 1920], device='cuda:0')
                c = a + b
                d = 2 * c + b
        """
        GPUMemory.__init__(self, _pid=_pid, _gpu_id=_gpu_id)
        multithread_.BaseThread.__init__(self)
        threading.Thread.__init__(self)
        self.limit_mem = _limit_mem
        self.unit_sec = _unit_sec
        self.pid = _pid
        self.gpu_id = _gpu_id
        self.is_print = _is_print
        self.mem_curr = 0.0
        self.is_lock = None
        self.is_terminate = None
        self._lock()
        self._unset_terminate()
        self.start()

    def __call__(self, _func):
        @wraps(_func)
        def decorator(*args, **kwargs):
            _func(*args, **kwargs)
            self._unlock()
            self.join()
        return decorator

    def run(self):
        while self.is_lock is True:
            self.mem_curr = self.get_mem_gpu_proc()
            if self.limit_mem <= self.mem_curr:
                if self.is_print is True:
                    print('The process [{}] will be terminated because the allocated GPU memory [{:.2f}] MiB '
                          'is exceeded its limit [{:.2f}] MiB.'.format(self.pid, self.mem_curr, self.limit_mem))
                self._unlock()
                self._set_terminate()
            time.sleep(self.unit_sec)

        if self.is_terminate is True:
            utils_.terminate_proc(_pid=self.pid)

    def _lock(self):
        self.is_lock = True

    def _unlock(self):
        self.is_lock = False

    def _set_terminate(self):
        self.is_terminate = True

    def _unset_terminate(self):
        self.is_terminate = False
