"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_xlsx.py
Description: A module for xlsx
"""

import openpyxl
import pandas as pd
from typing import Optional


class XLSX(object):
    def __init__(self, _spath_filename: str, _mode: str) -> None:
        super(XLSX, self).__init__()
        self.spath_filename = _spath_filename
        self.mode = _mode

        if _mode == 'w':
            self.wb = openpyxl.Workbook()
        elif _mode == 'r' or _mode == 'a':
            self.wb = openpyxl.load_workbook(filename=self.spath_filename)
        else:
            raise NotImplementedError('The _mode, {} may be incorrect.'.format(_mode))

        self.ws = self.wb.active

    def create_sheet(self, _title: Optional[str] = None, _index: Optional[int] = None) -> openpyxl.Workbook.worksheets:
        return self.wb.create_sheet(title=_title, index=_index)

    def save_workbook(self) -> None:
        self.wb.save(self.spath_filename)


def csv2xlsx(_spath_src: str, _spath_dst: str, _header: bool = True, _index: bool = True) -> None:
    df = pd.read_csv(_spath_src)
    df.to_excel(_spath_dst, header=_header, index=_index)