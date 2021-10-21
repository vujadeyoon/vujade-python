"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_txt.py
Description: A module for txt
"""


class TEXT(object):
    def __init__(self, _spath_filename: str, _mode: str) -> None:
        super(TEXT, self).__init__()
        self.spath_filename = _spath_filename
        self.mode = _mode

    def read_lines(self) -> list:
        with open(self.spath_filename, mode=self.mode) as f:
            lines = f.readlines()

        return lines

    def write_lines(self, _list_str: list) -> None:
        with open(self.spath_filename, mode=self.mode) as f:
            for _idx, _str in enumerate(_list_str):
                f.write(_str)

    def write(self, _str: str) -> None:
        with open(self.spath_filename, mode=self.mode) as f:
            f.write(_str)


def txt2dict(_spath_filename: str) -> dict:
    with open(_spath_filename, mode='r') as f:
        res = dict(line.rstrip().split(None, 1) for line in f)

    return res
