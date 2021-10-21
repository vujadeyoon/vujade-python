"""
from abc import *
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_multithread.py
Description: A module for multi-thread
"""


import abc


class BaseThread(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _lock(self):
        pass

    @abc.abstractmethod
    def _unlock(self):
        pass
