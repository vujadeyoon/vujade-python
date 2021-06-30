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
    def __init__(self, _spath: str):
        self.__spath = _spath
        self.name = self.path.stem
        self.ext = self.path.suffix
        self.filename = self.name + self.ext

    def __str__(self) -> str:
        return self.str

    def __repr__(self) -> str:
        return self.str

    @property
    def str(self) -> str:
        return self.__spath

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(self.str)

    @property
    def parent(self):
        return Path(_spath=str(self.path.parent))

    def replace_ext(self, _new: str):
        return Path(_spath=self.str.replace(self.ext, _new))

    def copy(self, _spath_dst: str) -> None:
        try:
            shutil.copy2(src=self.str, dst=_spath_dst)
        except Exception as e:
            raise OSError('The file copy is failed.: {}'.format(e))

    def unlink(self, _missing_ok: bool = True) -> None:
        if self.path.is_file() is True:
            self.path.unlink()
        else:
            if _missing_ok is False:
                raise FileNotFoundError('The file, {} is not existed.'.format(self.str))

    def cd(self) -> None:
        os.chdir(self.str)


def export_pythonpath(self, _spath: str) -> None:
    sys.path.append(_spath) # site.addsitedir(sitedir=_spath)


def uppath(_spath: str, _n: int = 1) -> str:
    return os.sep.join(_spath.split(os.sep)[:-_n])


def get_file_name_ext(_spath: str, _type_return: str = 'split') -> Union[str, Tuple[str, str]]:
    path_wo_filename = uppath(_spath=_spath, _n=1)
    file_name_ext = _spath[len(path_wo_filename) + 1:]
    if _type_return == 'join':
        return file_name_ext
    elif _type_return == 'split':
        filename, ext = os.path.splitext(file_name_ext)
        return filename, ext
    else:
        raise NotImplementedError


def get_glob(_spath: str, _file_ext: str) -> List[str]:
    return glob.glob('{}/*{}'.format(_spath.replace('[', '[[]'), _file_ext))
