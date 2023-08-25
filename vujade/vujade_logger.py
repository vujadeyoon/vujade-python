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
import traceback
import warnings
import logging
import logging.handlers
from typing import Optional
from datetime import datetime
from pytz import timezone
from colorlog import ColoredFormatter
from vujade import vujade_path as path_
from vujade import vujade_str as str_
from vujade import vujade_utils as utils_


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


class _BaseSimpleLog(object):
    logger = logging.getLogger(__name__)

    @staticmethod
    def get_spath_log() -> Optional[str]:
        path_log = path_.Path(utils_.get_env_var(_name_var='PATH_LOG', _default='', _is_raise_existed=False))

        if path_log.ext in {'.log', '.txt'}:
            res = '{}'.format(path_log)
        else:
            res = None

        return res

    @staticmethod
    def get_level_log() -> int:
        level_log = utils_.get_env_var(_name_var='LEVEL_LOG', _default='DEBUG', _is_raise_existed=False)

        try:
            res = getattr(logging, level_log)
        except Exception as e:
            res = getattr(logging, 'DEBUG')

        return res

    @staticmethod
    def get_is_traceback_print_stack() -> bool:
        try:
            res = str_.str2bool(utils_.get_env_var(_name_var='IS_TRACEBACK_PRINT_STACK', _default='False', _is_raise_existed=False))
        except Exception as e:
            res = False

        return res

    @staticmethod
    def _timetz(*args):
        tz = timezone('Asia/Seoul') # UTC; Asia/Seoul; Asia/Shanghai; Europe/Berlin
        return datetime.now(tz).timetuple()

    @classmethod
    def _warning_handler(cls, _message, _category, _filename, _lineno, _file=None, _line=None) -> None:
        info_trace = f'[{_filename}:{_lineno}] {_category.__name__}: {_message}'
        if cls.get_is_traceback_print_stack() is True:
            traceback.print_stack()

        cls.logger.warning(info_trace)


class SimpleLog(object):
    """
    Usage:
        export PATH_LOG='./log/debug.log'
        from vujade import vujade_logger as loggger_
        loggger_.SimpleLog.d(_tag='TAG', _message='MESSAGE')
    """

    var_log = {
        'path': _BaseSimpleLog.get_spath_log(),
        'level': _BaseSimpleLog.get_level_log(),
    }

    debug_info = DEBUG()

    if var_log['path'] is not None:
        path_log = path_.Path(var_log['path'])
        path_log.parent.path.mkdir(mode=0o775, parents=True, exist_ok=True)

        _BaseSimpleLog.logger.setLevel(var_log['level'])

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
        streamHandler.setLevel(var_log['level'])
        streamHandler.setFormatter(formatter_color)
        _BaseSimpleLog.logger.addHandler(streamHandler)

        fileHandler = logging.FileHandler(path_log.str)
        fileHandler.setLevel(var_log['level'])
        fileHandler.setFormatter(formatter)
        _BaseSimpleLog.logger.addHandler(fileHandler)

        warnings.showwarning = _BaseSimpleLog._warning_handler

    @classmethod
    def _check_valid_path_log(cls):
        if cls.var_log['path'] is None:
            raise ValueError('The environment variable, PATH_LOG (i.e. *.log and *.txt) should be assigned correctly in advance in order to use the class, SimpleLog.')

    @classmethod
    def _get_message(cls, _tag: Optional[str], _message: str) -> str:
        cls.debug_info.get_file_line()
        res = '[{}:{}] '.format(cls.debug_info.fileName, cls.debug_info.lineNumber)
        res += '{}'.format(_message) if _tag is None else '[{}] {}'.format(_tag, _message)

        return res

    @classmethod
    def d(cls, _tag: Optional[str], _message: str) -> None:
        cls._check_valid_path_log()
        _BaseSimpleLog.logger.debug(cls._get_message(_tag=_tag, _message=_message))

    @classmethod
    def i(cls, _tag: Optional[str], _message: str) -> None:
        cls._check_valid_path_log()
        _BaseSimpleLog.logger.info(cls._get_message(_tag=_tag, _message=_message))

    @classmethod
    def w(cls, _tag: Optional[str], _message: str) -> None:
        cls._check_valid_path_log()
        _BaseSimpleLog.logger.warning(cls._get_message(_tag=_tag, _message=_message))

    @classmethod
    def e(cls, _tag: Optional[str], _message: str) -> None:
        cls._check_valid_path_log()
        _BaseSimpleLog.logger.error(cls._get_message(_tag=_tag, _message=_message))

    @classmethod
    def c(cls, _tag: Optional[str], _message: str) -> None:
        cls._check_valid_path_log()
        _BaseSimpleLog.logger.critical(cls._get_message(_tag=_tag, _message=_message))


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
        self.logger_warnings = logging.getLogger('py.warnings')
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
