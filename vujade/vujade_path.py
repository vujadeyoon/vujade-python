"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_path.py
Description: A module for path
"""


import os
import glob
import site
import pathlib
from typing import Union, List, Tuple


class Path(object):
    def __init__(self, _path: str):
        self.__path_str = _path
        self.__path = pathlib.Path(_path)
        self.parent = self.path.parent
        self.name = self.path.stem
        self.ext = self.path.suffix

    def __str__(self) -> str:
        return self.str

    def __repr__(self) -> str:
        return self.str

    @property
    def path(self) -> pathlib.Path:
        return self.__path

    @property
    def str(self) -> str:
        return self.__path_str

    def export_pythonpath(self) -> None:
        site.addsitedir(sitedir=self.path)

    def replace_ext(self, _new):
        return Path(_path=self.str.replace(self.ext, _new))


def uppath(_path: str, _n: int = 1) -> str:
    return os.sep.join(_path.split(os.sep)[:-_n])


def get_file_name_ext(_path: str, _type_return: str = 'split') -> Union[str, Tuple[str, str]]:
    path_wo_filename = uppath(_path=_path, _n=1)
    file_name_ext = _path[len(path_wo_filename) + 1:]
    if _type_return == 'join':
        return file_name_ext
    elif _type_return == 'split':
        filename, ext = os.path.splitext(file_name_ext)
        return filename, ext
    else:
        raise NotImplementedError


def get_glob(_path: str, _file_ext: str) -> List[str]:
    return glob.glob('{}/*{}'.format(_path.replace('[', '[[]'), _file_ext))
