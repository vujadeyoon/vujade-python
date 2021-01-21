"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_txt.py
Description: A module for txt
"""


class vujade_txt:
    def __init__(self, _path_filename: str):
        self.path_filename = _path_filename

    def read_lines(self):
        with open(self.path_filename) as f:
            lines = f.readlines()

        return lines
