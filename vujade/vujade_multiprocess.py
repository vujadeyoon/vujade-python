"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_multiprocess.py
Description: A module for multi-processing
"""


import os
import time
from multiprocessing import Process
from multiprocessing import Queue


class BaseMultiProcess(object):
    def __init__(self, _target_method, _num_proc: int = os.cpu_count()) -> None:
        super(BaseMultiProcess, self).__init__()
        self.target_method = _target_method
        self.num_proc = _num_proc

    def _proc_setup(self) -> None:
        self.queue = Queue()
        self.process = [Process(target=self.target_method, args=(self.queue,)) for _ in range(self.num_proc)]

        for p in self.process: p.start()

    def _proc_release(self) -> None:
        for _ in range(self.num_proc): self.queue.put((None, None)) # Todo: The number of parameters for queue.put() should be checked.
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()
