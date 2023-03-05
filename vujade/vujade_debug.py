"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_debug.py
Description: A module for debug
"""


import os
import re
import traceback


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


def printf(*_args, **_kwargs) -> str:
    debug_info = DEBUG()
    debug_info.get_file_line()

    info_str = ''
    for _idx, _arg in enumerate(_args):
        info_str += '{} '.format(_arg)
    info_str = info_str.rstrip(' ')

    info_trace = '[{}:{}] '.format(debug_info.fileName, debug_info.lineNumber) + info_str

    if ('_is_pause' in _kwargs.keys()) and (_kwargs['_is_pause'] is False):
        _print = print
    else:
        _print = input

    if ('_is_print' in _kwargs.keys()) and (_kwargs['_is_print'] is False):
        pass
    else:
        _print(info_trace)

    return info_trace
