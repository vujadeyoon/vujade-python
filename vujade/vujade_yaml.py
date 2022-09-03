"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_videocv.py
Description: A module for yaml.
"""


import yaml


class YAML(object):
    def __init__(self, _spath_filename: str) -> None:
        super(YAML, self).__init__()
        self.spath_filename = _spath_filename

    def read(self) -> None:
        with open(self.spath_filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        return data

    def write(self, _data: dict) -> None:
        with open(self.spath_filename, 'w') as f:
            yaml.dump(_data, f)
