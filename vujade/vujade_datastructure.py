"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_datastructure.py
Description: A collection of data structure
"""


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)


class Stack(list):
    """
    Usage:
        1) s = Stack()
        2) s.push(1)
        3) s.pop()
    """

    def __init__(self, _init_list=None):
        super(Stack, self).__init__()
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


class Queue(list):
    """
    Usage:
        1) q = Queue()
        2) q.enqueue()
        3) q.dequeue()
    """

    def __init__(self, _init_list=None):
        super(Queue, self).__init__()
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