"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_bytes.py
Description: A module for bytes
"""


class Bytes(object):
    @staticmethod
    def read(_spath_file: str) -> bytes:
        with open(_spath_file, 'rb') as f:
            res = f.read()
        return res

    @staticmethod
    def write(_spath_file: str, _bytes: bytes) -> None:
        with open(_spath_file, 'wb') as f:
            f.write(_bytes)
