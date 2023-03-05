"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_logger.py
Description: A module for logger
"""


import os
import sys
import re
import builtins
import traceback
import warnings
import logging
import logging.handlers
import vujade
from typing import Optional
from datetime import datetime
from pytz import timezone
from colorlog import ColoredFormatter
from vujade import vujade_path as path_
from vujade import vujade_warnings as warnings_


builtin_func = dict()
for _idx, _name_builtin in enumerate(vujade.env_var['log']['builtins']):
    builtin_func[_name_builtin] = getattr(builtins, _name_builtin)


class DEBUG(object):
    def __init__(self):
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


def print_with_log(*args, **kwargs) -> None:
    """
    Usage:
        from vujade.vujade_logger import print_with_log as print
        print('Test')
        print('Test', _tag='Tag', _type='i')
    """
    types = {'d', 'i', 'w', 'e', 'c'}

    if ('_tag' in kwargs.keys()) and (isinstance(kwargs['_tag'], str) is True):
        log_tag = kwargs['_tag']
    else:
        log_tag = None

    if ('_type' in kwargs.keys()) and (isinstance(kwargs['_type'], str) is True) and (kwargs['_type'] in types):
        log_type = kwargs['_type']
    else:
        log_type = 'i'

    debug_info = DEBUG()
    debug_info.get_file_line()

    log_str = ''
    for _idx, _arg in enumerate(args):
        log_str += '{} '.format(_arg)
    log_str = log_str.rstrip(' ')

    if (vujade.env_var['log']['path'] is not None) and (debug_info.fileName == 'vujade_debug.py'):
        info_trace = log_str
    else:
        info_trace = '[{}:{}] '.format(debug_info.fileName, debug_info.lineNumber) + log_str

    if vujade.env_var['log']['path'] is not None:
        attr = getattr(SimpleLog, log_type)
        attr(_tag=log_tag, _message=info_trace)
    else:
        attr = getattr(builtins, 'print')
        attr(log_str)


class _BaseSimpleLog(object):
    logger = logging.getLogger(__name__)

    @staticmethod
    def _timetz(*args):
        tz = timezone('Asia/Seoul') # UTC; Asia/Seoul; Asia/Shanghai; Europe/Berlin
        return datetime.now(tz).timetuple()

    @classmethod
    def _warning_handler(cls, _message, _category, _filename, _lineno, _file=None, _line=None) -> None:
        info_trace = f'[{_filename}:{_lineno}] {_category.__name__}: {_message}'
        if vujade.env_var['log']['is_traceback_print_stack'] is True:
            traceback.print_stack()

        cls.logger.warning(info_trace)

    @staticmethod
    def _update_builtins() -> None:
        for _idx, _name_builtin in enumerate(vujade.env_var['log']['builtins']):
            setattr(builtins, _name_builtin, print_with_log)

    @staticmethod
    def _restore_builtins() -> None:
        for _idx, _name_builtin in enumerate(vujade.env_var['log']['builtins']):
            setattr(builtins, _name_builtin, builtin_func[_name_builtin])


class SimpleLog(object):
    if vujade.env_var['log']['path'] is not None:
        path_log = path_.Path(vujade.env_var['log']['path'])
        path_log.parent.path.mkdir(mode=0o775, parents=True, exist_ok=True)

        _BaseSimpleLog.logger.setLevel(vujade.env_var['log']['level'])

        logging.Formatter.converter = _BaseSimpleLog._timetz
        fmt = '%(log_color)s[%(asctime)s] [%(levelname)s (%(process)s)]: %(message)s'
        log_colors = {
            'INFO': '',
            'DEBUG': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        }
        formatter_color = ColoredFormatter(fmt=fmt, log_colors=log_colors)
        formatter = ColoredFormatter(fmt=fmt, log_colors=log_colors, no_color=True)

        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(vujade.env_var['log']['level'])
        streamHandler.setFormatter(formatter_color)
        _BaseSimpleLog.logger.addHandler(streamHandler)

        if vujade.env_var['verbose']['level'] in {2, 3}:
            fileHandler = logging.FileHandler(path_log.str)
            fileHandler.setLevel(vujade.env_var['log']['level'])
            fileHandler.setFormatter(formatter)
            _BaseSimpleLog.logger.addHandler(fileHandler)

        warnings.showwarning = _BaseSimpleLog._warning_handler

        if vujade.env_var['log']['builtins']:
            _BaseSimpleLog._update_builtins()

    def __del__(self):
        if vujade.env_var['log']['builtins']:
            _BaseSimpleLog._restore_builtins()

    @classmethod
    def d(cls, _tag: Optional[str], _message: str) -> None:
        _BaseSimpleLog.logger.debug('{}'.format(_message) if _tag is None else '[{}] {}'.format(_tag, _message))

    @classmethod
    def i(cls, _tag: Optional[str], _message: str) -> None:
        _BaseSimpleLog.logger.info('{}'.format(_message) if _tag is None else '[{}] {}'.format(_tag, _message))

    @classmethod
    def w(cls, _tag: Optional[str], _message: str) -> None:
        _BaseSimpleLog.logger.warning('{}'.format(_message) if _tag is None else '[{}] {}'.format(_tag, _message))

    @classmethod
    def e(cls, _tag: Optional[str], _message: str) -> None:
        _BaseSimpleLog.logger.error('{}'.format(_message) if _tag is None else '[{}] {}'.format(_tag, _message))

    @classmethod
    def c(cls, _tag: Optional[str], _message: str) -> None:
        _BaseSimpleLog.logger.critical('{}'.format(_message) if _tag is None else '[{}] {}'.format(_tag, _message))


class Logger(object):
    def __init__(self, _path_log, _mode='a', _fmt='[%(asctime)s] [%(levelname)s (%(process)s)] [%(name)s]: %(message)s', _level=logging.DEBUG):
        super(Logger, self).__init__()
        self.path_log = _path_log
        self.mode = _mode
        self.fmt = _fmt
        self.level = _level
        self.log_colors = {
            'INFO': '',
            'DEBUG': 'cyan',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        }

    def get_logger(self):
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(level=self.level)
        self.logger_warnings = logging.getLogger("py.warnings")
        self._set_handler()

        return self.logger

    def _set_handler(self):
        logging.captureWarnings(True)

        if self.fmt is not None:
            formatter_file = logging.Formatter(fmt=self.fmt)
            formatter_stream = ColoredFormatter(fmt='%(log_color)s' + self.fmt, log_colors=self.log_colors)

        fileHandler = logging.FileHandler(filename=self.path_log, mode=self.mode)
        streamHandler = logging.StreamHandler()

        if self.fmt is not None:
            fileHandler.setFormatter(fmt=formatter_file)
            streamHandler.setFormatter(fmt=formatter_stream)

        self.logger.addHandler(hdlr=fileHandler)
        self.logger.addHandler(hdlr=streamHandler)
        self.logger_warnings.addHandler(hdlr=fileHandler)
        self.logger_warnings.addHandler(hdlr=streamHandler)


class Print2Logger(object):
    def __init__(self, _path_log, _mode='a'):
        super(Print2Logger, self).__init__()
        self.path_log = _path_log
        self.mode = _mode
        self._run()

    def __del__(self):
        self.close()

    def info(self, _str):
        print(_str)

    def close(self):
        sys.stdout = self.stdout_ori
        self.fp_log.close()

    def _run(self):
        self.stdout_ori = sys.stdout
        self.fp_log = open(self.path_log, self.mode)
        sys.stdout = self.fp_log
