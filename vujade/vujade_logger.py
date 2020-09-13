"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Sep. 13, 2020.

Title: vujade_logger.py
Version: 0.1
Description: A module for logger
"""


import logging
import logging.handlers
from colorlog import ColoredFormatter

class vujade_logger():
    def __init__(self, _filename, _mode='a', _fmt='[%(asctime)s] [%(levelname)s (%(process)s)]: %(message)s', _level=logging.DEBUG):
        super(vujade_logger, self).__init__()
        self.filename = _filename
        self.mode = _mode
        self.fmt = _fmt
        self.level = _level
        self.log_colors = {
                'DEBUG': 'cyan',
                'INFO': 'white,bold',
                'INFOV': 'cyan,bold',
                'WARNING': 'yellow',
                'ERROR': 'red,bold',
                'CRITICAL': 'red,bg_white',
            }

    def get_logger(self):
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(level=self.level)
        self._set_handler()

        return self.logger

    def _set_handler(self):
        formatter_file = logging.Formatter(fmt=self.fmt)
        formatter_stream = ColoredFormatter(fmt='%(log_color)s' + self.fmt, log_colors=self.log_colors)

        fileHandler = logging.FileHandler(filename=self.filename, mode=self.mode)
        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(fmt=formatter_file)
        streamHandler.setFormatter(fmt=formatter_stream)

        if not self.logger.handlers:
            self.logger.addHandler(hdlr=fileHandler)
            self.logger.addHandler(hdlr=streamHandler)
