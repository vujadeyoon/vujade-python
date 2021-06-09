"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_download.py
Description: A module for download
"""


import requests
import shutil


class vujade_download:
    def __init__(self):
        pass

    def run(self, _url: str, _path_filename: str) -> bool:
        r = requests.get(_url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True

            with open(_path_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

            res = True
        else:
            res = False

        return res