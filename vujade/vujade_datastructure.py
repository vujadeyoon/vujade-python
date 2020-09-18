"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Sep. 13, 2020.

Title: vujade_datastructure.py
Version: 0.1
Description: A collection of data structure
"""


class stack(list):
    """
    Usage:
        1) s = Stack()
        2) s.push(1)
        3) s.pop()
    """

    def __init__(self, _init_list=None):
        super(stack, self).__init__()
        if _init_list is not None:
            self._init(_list=_init_list)

    def _init(self, _list):
        for idx in range(len(_list)):
            self.push(_element=_list[idx])

    def push(self, _element):
        self.append(_element)

    def is_empty(self):
        if not self:
            return True
        else:
            return False

    def peek(self):
        return self[-1]


class queue(list):
    """
    Usage:
        1) q = Queue()
        2) q.enqueue()
        3) q.dequeue()
    """

    def __init__(self, _init_list=None):
        super(queue, self).__init__()
        if _init_list is not None:
            self._init(_list=_init_list)

    def _init(self, _list):
        for idx in range(len(_list)):
            self.enqueue(_element=_list[idx])

    def enqueue(self, _element):
        self.append(_element)

    def dequeue(self):
        return self.pop(0)

    def is_empty(self):
        if not self:
            return True
        else:
            return False

    def peek(self):
        return self[0]