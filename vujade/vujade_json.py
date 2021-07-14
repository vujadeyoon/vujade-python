"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_json.py
Description: A module for json
"""


import json


class JSON(object):
    def __init__(self, _spath_filename: str, _mode: str):
        super(JSON, self).__init__()
        self.spath_filename = _spath_filename
        self.mode = _mode

    def read(self) -> dict:
        with open(self.spath_filename, self.mode) as f:
            data = json.load(f)

        return data

    def write(self, _dict_data: dict, _indent: int = 4, _ensure_ascii: bool = True) -> None:
        with open(self.spath_filename, self.mode) as f:
            json.dump(_dict_data, f, indent=_indent, ensure_ascii=_ensure_ascii)

    def pprint(self) -> None:
        print(json.dumps(self.read(), indent=2, sort_keys=True))
