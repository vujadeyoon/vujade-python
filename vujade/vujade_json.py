"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_json.py
Description: A module for json
"""


import json


class vujade_json():
    def __init__(self, _path_filename: str, _mode: str):
        super(vujade_json, self).__init__()
        self.path_filename = _path_filename
        self.mode = _mode

    def read(self) -> dict:
        with open(self.path_filename, self.mode) as f:
            data = json.load(f)

        return data

    def write(self, _dict_data: dict, _indent: int = 4) -> None:
        with open(self.path_filename, self.mode) as f:
            json.dump(_dict_data, f, indent=_indent)

    def pprint(self) -> None:
        print(json.dumps(self.read(), indent=2, sort_keys=True))
