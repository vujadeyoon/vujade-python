"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Sep. 21, 2020.

Title: vujade_videocv.py
Version: 0.1.0
Description: A module for yaml.
"""


import yaml

class vujade_yaml():
    def __init__(self, _path_filename):
        super(vujade_yaml, self).__init__()
        self.path_filename = _path_filename

    def read(self):
        with open(self.path_filename) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        return data
