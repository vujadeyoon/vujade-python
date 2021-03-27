"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_txt.py
Description: A module for txt
"""


class vujade_txt:
    def __init__(self, _path_filename: str, _mode: str) -> None:
        self.path_filename = _path_filename
        self.mode = _mode

    def read_lines(self) -> list:
        with open(self.path_filename, mode=self.mode) as f:
            lines = f.readlines()

        return lines

    def write(self, _str: str) -> None:
        with open(self.path_filename, mode=self.mode) as f:
            f.write(_str)


def txt2dict(_path_txt: str) -> dict:
    with open(_path_txt, mode='r') as f:
        res = dict(line.rstrip().split(None, 1) for line in f)

    return res
