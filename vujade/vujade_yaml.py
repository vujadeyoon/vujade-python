"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_videocv.py
Description: A module for yaml.
"""


import yaml


class YAML(object):
    @staticmethod
    def read(_spath_filename: str, _mode: str = 'r') -> dict:
        with open(_spath_filename, _mode) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        return data

    @staticmethod
    def write(_spath_filename: str, _data: dict, _mode: str = 'w') -> None:
        with open(_spath_filename, _mode) as f:
            yaml.dump(_data, f)
