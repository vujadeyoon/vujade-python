"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_multithread.py
Description: A module for multi-thread
"""


from abc import *


class ThreadBase(metaclass=ABCMeta):
    @abstractmethod
    def _lock(self):
        pass

    @abstractmethod
    def _unlock(self):
        pass
