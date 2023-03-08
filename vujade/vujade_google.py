"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_google.py
Description: A module for Google Cloud API.

Reference:
    i)   https://developers.google.com/sheets/api
    ii)  https://velog.io/@junsugi/Google-Sheet-연동하기-feat.-Google-APIi
    iii) https://jsikim1.tistory.com/205
    iv)  https://velog.io/@jmon/구글시트-API-를-이용한-읽고-쓰기-Google-SpreadSheets-API-JMON
"""


import gspread
from typing import Optional, Tuple
from oauth2client.service_account import ServiceAccountCredentials
from vujade import vujade_str as str_
from vujade.vujade_debug import printd


class GoogleWorkSheet(object):
    def __init__(self, _spreadsheet: gspread.spreadsheet.Spreadsheet, _name_worksheet: str = 'Sheet1') -> None:
        super(GoogleWorkSheet, self).__init__()
        self.worksheet = _spreadsheet.worksheet(title=_name_worksheet)

    def get_all_values(self) -> list:
        return self.worksheet.get_all_values()

    def read_value(self, _name_cell: str) -> str:
        return self.worksheet.acell(label=_name_cell).value

    def write_value(self, _name_cell: str, _value: str) -> None:
        self.worksheet.update_acell(label=_name_cell, value=_value)


class GoolgeSheet(object):
    def __init__(self, _spath_json_key: str, _key_spreadsheet: str) -> None:
        super(GoolgeSheet, self).__init__()
        self.spath_json_key = _spath_json_key
        self.key_spreadsheet = _key_spreadsheet
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        self.credential = ServiceAccountCredentials.from_json_keyfile_name(self.spath_json_key, self.scope)
        self.gc = gspread.authorize(self.credential)
        self.spreadsheet = self.gc.open_by_key(self.key_spreadsheet)
        self.cache_worksheet = dict()

    def worksheet(self, _name_worksheet: str = 'Sheet1') -> GoogleWorkSheet:
        if not _name_worksheet in self.cache_worksheet.keys():
            self.cache_worksheet[_name_worksheet] = GoogleWorkSheet(_spreadsheet=self.spreadsheet, _name_worksheet=_name_worksheet)

        return self.cache_worksheet[_name_worksheet]


class GoogleSheet3DMM(object):
    def __init__(self, _spath_json_key: str, _key_spreadsheet: str, _name_worksheet: str = 'AFLW2000-3D-FaicalAlignment'):
        super(GoogleSheet3DMM, self).__init__()
        self.spath_json_key = _spath_json_key
        self.key_spreadsheet = _key_spreadsheet
        self.name_worksheet = _name_worksheet
        self.gs = GoolgeSheet(_spath_json_key=self.spath_json_key, _key_spreadsheet=self.key_spreadsheet)
        self.ws = self.gs.worksheet(_name_worksheet=_name_worksheet)
        self.data_facial_alignment = {
            '0_to_30': 0.0,
            '30_to_60': 0.0,
            '60_to_90': 0.0,
            'All': 0.0,
            'Description': ''
        }
        self.idx_write = len(self.ws.get_all_values()) + 1

    def _get_idx_write(self) -> int:
        return self.idx_write

    def _update_idx_write(self):
        self.idx_write += 1

    def _get_idx_keyword(self, _list_all_values: list, _keyword: str = 'Algorithm') -> int:
        res = -1
        for _idx, _row in enumerate(_list_all_values):
            if _row[0] == _keyword:
                res = _idx + 1
                break

        return res

    def get_data_facial_alignment(self) -> dict:
        res = dict()
        list_all_values = self.ws.get_all_values()

        idx_start = self._get_idx_keyword(_list_all_values=list_all_values, _keyword='Algorithm')
        for _idx, _row in enumerate(list_all_values[idx_start:]):
            try:
                self.data_facial_alignment['0_to_30'] = float(_row[1])
                self.data_facial_alignment['30_to_60'] = float(_row[2])
                self.data_facial_alignment['60_to_90'] = float(_row[3])
                self.data_facial_alignment['All'] = float(_row[4])
                self.data_facial_alignment['Description'] = _row[5]
                res[_row[0]] = self.data_facial_alignment
            except Exception as e:
                raise ValueError(e)

        return res

    def set_data_facial_alignment(self, _data_facial_alignment: dict, _columns: Tuple[str, str]) -> None:
        idx_column = str_.get_alphabets(_columns=_columns)

        for _idx, (_column, (_key, _val)) in enumerate(zip(idx_column, _data_facial_alignment.items())):
            gs_3dmm.ws.write_value('{}'.format(_column + str(self._get_idx_write())), str(_val))
        self._update_idx_write()


if __name__=='__main__':
    spath_json_key = '/home/sjyoon1671/Desktop/dmmtransformer-377b3096e465.json'
    key_spreadsheet = '1hcNdlzFY0xfiyezdSpBen6GrRqVxfMrjk_sVpxndu3A'
    name_worksheet = 'AFLW2000-3D-FaicalAlignment'

    gs_3dmm = GoogleSheet3DMM(_spath_json_key=spath_json_key, _key_spreadsheet=key_spreadsheet, _name_worksheet=name_worksheet)

    data_facial_alignment_curr = {
        'run_id': 'run_id',
        '0_to_30': 1.11,
        '30_to_60': 2.11,
        '60_to_90': 3.11,
        'All': 4.11,
        'Description': 'FirstA+'
    }

    # printd(gs_3dmm.get_data_facial_alignment())
    gs_3dmm.set_data_facial_alignment(_data_facial_alignment=data_facial_alignment_curr, _columns=('A', 'F'))
