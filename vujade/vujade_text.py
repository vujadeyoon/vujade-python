"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_txt.py
Description: A module for txt
"""


class TEXT(object):
    @staticmethod
    def read_lines(_spath_filename: str) -> list:
        with open(_spath_filename, 'r') as f:
            lines = f.readlines()

        return lines

    @staticmethod
    def write_lines(self, _spath_filename: str, _list_str: list, _mode: str = 'w') -> None:
        with open(_spath_filename, mode=_mode) as f:
            for _idx, _str in enumerate(_list_str):
                f.write(_str)

    @staticmethod
    def write(self, _spath_filename: str, _str: str, _mode: str = 'w') -> None:
        with open(_spath_filename, mode=_mode) as f:
            f.write(_str)


def txt2dict(_spath_filename: str) -> dict:
    with open(_spath_filename, mode='r') as f:
        res = dict(line.rstrip().split(None, 1) for line in f)

    return res
