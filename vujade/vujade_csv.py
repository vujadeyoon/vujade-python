"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_csv.py
Description: A module for csv
"""


import os
import pandas as pd

class vujade_csv():
    def __init__(self, _path_filename, _isremove=False, _header=None, _index=False, _mode_write='a'):
        super(vujade_csv, self).__init__()
        self.path_filename = _path_filename
        self.header = _header
        self.index = _index
        self.mode_write = _mode_write

        self.isFirstWrite = True

        if _isremove is True and os.path.isfile(path=_path_filename):
            os.remove(path=_path_filename)

    def read(self):
        return pd.read_csv(self.path_filename)

    def write(self, _ndarr):
        if self.isFirstWrite is True:
            pd.DataFrame(_ndarr).to_csv(self.path_filename, header=self.header, index=self.index, mode=self.mode_write)
            self.isFirstWrite = False
        else:
            pd.DataFrame(_ndarr).to_csv(self.path_filename, header=False, index=self.index, mode=self.mode_write)