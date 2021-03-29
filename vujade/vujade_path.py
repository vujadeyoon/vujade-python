"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_path.py
Description: A module for path
"""


import site


def export_pythonpath(_path: str) -> None:
    site.addsitedir(sitedir=_path)
