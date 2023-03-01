"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_download.py
Description: A module for download
"""


import requests
import shutil


class Download(object):
    def __init__(self):
        super(Download, self).__init__()
        pass

    @staticmethod
    def run(_url: str, _spath_filename: str) -> bool:
        r = requests.get(_url, stream=True)

        if r.status_code == 200:
            r.raw.decode_content = True

            with open(_spath_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)

            res = True
        else:
            res = False

        return res
