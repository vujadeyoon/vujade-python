"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_json.py
Description: A module for json
"""


import json


class JSON(object):
    @staticmethod
    def read(_spath_filename: str, _mode: str = 'r') -> dict:
        with open(_spath_filename, _mode) as f:
            data = json.load(f)

        return data

    @staticmethod
    def write(_spath_filename: str, _data: dict, _indent: int = 4, _ensure_ascii: bool = True, _mode: str = 'w') -> None:
        with open(_spath_filename, _mode) as f:
            json.dump(_data, f, indent=_indent, ensure_ascii=_ensure_ascii)

    @classmethod
    def pretty_read(cls, _spath_filename: str, _ensure_ascii: bool = True, _mode: str = 'r') -> str:
        return json.dumps(
            cls.read(_spath_filename=_spath_filename, _mode=_mode),
            indent=2,
            sort_keys=True,
            ensure_ascii=_ensure_ascii
        )