"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_videocv.py
Description: A module for yaml.
"""


import yaml


class YAML(object):
    def __init__(self, _spath_filename: str, _mode: str) -> None:
        super(YAML, self).__init__()
        self.spath_filename = _spath_filename
        self.mode = _mode

    def read(self) -> dict:
        with open(self.spath_filename, self.mode) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        return data

    def write(self, _data: dict) -> None:
        with open(self.spath_filename, self.mode) as f:
            yaml.dump(_data, f)
