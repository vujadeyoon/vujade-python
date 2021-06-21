"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_path.py
Description: A module for path
"""


import os
import glob
import sys
import pathlib
import shutil
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

    def replace_ext(self, _new: str):
        return Path(_path=self.str.replace(self.ext, _new))

    def copy(self, _dst: str) -> None:
        try:
            shutil.copy2(src=self.str, dst=_dst)
        except Exception as e:
            raise OSError('The file copy is failed.: {}'.format(e))

    def unlink(self, _missing_ok: bool = True) -> None:
        if self.path.is_file() is True:
            self.path.unlink()
        else:
            if _missing_ok is False:
                raise FileNotFoundError('The file, {} is not existed.'.format(self.str))


def export_pythonpath(self, _path: str) -> None:
    sys.path.append(_path) # site.addsitedir(sitedir=_path)


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
