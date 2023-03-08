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

    @staticmethod
    def get_colors() -> dict:
        return {
            'PUPPLE': '\033[95m',
            'BLUE': '\033[94m',
            'GREEN': '\033[92m',
            'RED': '\033[91m',
            'YELLOW': '\033[93m',
            'FATAL': '\033[91m',
            'WARNING': '\033[93m',
            'ENDC': '\033[0m',
            'BOLD': '\033[1m',
            'UNDERLINE': '\033[4m'
        }

    @staticmethod
    def get_key_color_default() -> str:
        return 'YELLOW'


def encode_color(*_args, **_kwargs) -> str:
    colors = DEBUG.get_colors()

    # Assign default color.
    if ('_color' not in _kwargs.keys()):
        _kwargs['_color'] = DEBUG.get_key_color_default()

    if ('_color' in _kwargs.keys()) and (_kwargs['_color'] not in colors):
        raise ValueError('The given color, {} should be included in colors (i.e. {}).'.format(_kwargs['_color'], colors.keys()))

    info_str = ''
    for _idx, _arg in enumerate(_args):
        info_str += '{} '.format(_arg)
    info_str = info_str.rstrip(' ')

    if ('_color' in _kwargs.keys()) and (_kwargs['_color'] in colors):
        info_colored = colors[_kwargs['_color']] + info_str + colors['ENDC']
    else:
        raise ValueError

    return info_colored


def printd(*_args, **_kwargs) -> str:
    debug_info = DEBUG()
    debug_info.get_file_line()

    info_str = ''
    for _idx, _arg in enumerate(_args):
        info_str += '{} '.format(_arg)
    info_str = info_str.rstrip(' ')

    info_traced = '[{}:{}] '.format(debug_info.fileName, debug_info.lineNumber) + info_str

    info_traced_colored = encode_color(info_traced, **_kwargs)

    if ('_is_pause' in _kwargs.keys()) and (_kwargs['_is_pause'] is False):
        _print = print
    else:
        _print = input

    if ('_is_print' in _kwargs.keys()) and (_kwargs['_is_print'] is False):
        pass
    else:
        _print(info_traced_colored)

    return info_traced_colored
