"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_csv.py
Description: A module for csv
"""


import numpy as np
import pandas as pd
from typing import Optional
from vujade import vujade_path as path_


class CSV(object):
    def __init__(self, _spath_filename: str, _is_remove: bool = False, _header: Optional[list] = None, _index: bool = False, _mode_write: str = 'a'):
        super(CSV, self).__init__()
        self.path_filename = path_.Path(_spath=_spath_filename)
        self.header = _header
        self.index = _index
        self.mode_write = _mode_write
        self.isFirstWrite = True

        if _is_remove is True and self.path_filename.path.is_file() is True:
            self.path_filename.path.unlink()

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path_filename.str)

    def write(self, _ndarr: np.array) -> None:
        if self.isFirstWrite is True:
            pd.DataFrame(_ndarr).to_csv(self.path_filename.str, header=self.header, index=self.index, mode=self.mode_write)
            self.isFirstWrite = False
        else:
            pd.DataFrame(_ndarr).to_csv(self.path_filename.str, header=False, index=self.index, mode=self.mode_write)
