"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Sep. 13, 2020.

Title: vujade_json.py
Version: 0.1
Description: A module for json
"""


import json

class vujade_json():
    def __init__(self, _path_filename):
        super(vujade_json, self).__init__()
        self.path_filename = _path_filename

    def read(self):
        with open(self.path_filename, 'r') as f:
            data = json.load(f)

        return data

    def write(self, _dict_data, _indent=4):
        with open(self.path_filename, 'w') as f:
            json.dump(_dict_data, f, indent=_indent)

    def pprint(self):
        print(json.dumps(self.read(), indent=2, sort_keys=True))