"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_compression.py
Description: A module for compression
"""


import os
import abc
import zipfile
from typing import List, Optional
from vujade import vujade_path as path_
from vujade import vujade_list as list_


class Compression(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compress(self) -> None:
        pass

    @abc.abstractmethod
    def decompress(self) -> None:
        pass


class Zip(object):
    def __init__(self, _pwd: Optional[bytes] = None) -> None:
        super(Zip, self).__init__()
        if (_pwd is None) or (isinstance(_pwd, bytes) is True):
            self.path_cwd = path_.Path(_spath=os.getcwd())
            self.pwd = _pwd
        else:
            raise ValueError('The type of _pwd, {} may be incorrect.'.format(type(_pwd)))

    def compress(self, _spath_zip: str, _targets: List[str], _spath_base: Optional[str] = None, _level: int = zipfile.ZIP_DEFLATED) -> None:
        if _spath_base is None:
            self._compress_full(_spath_zip=_spath_zip, _targets=_targets, _level=_level)
        else:
            self._compress_compact(_spath_zip=_spath_zip, _targets=_targets, _spath_base=_spath_base, _level=_level)

    def decompress(self, _spath_zip: str, _d: Optional[str] = None) -> None:
        path_zip = path_.Path(_spath=_spath_zip)

        if _d is not None:
            path_base = path_.Path(_spath=os.path.join(path_zip.parent.str, _d))
            path_zip_dst = path_.Path(_spath=os.path.join(path_base.str, path_zip.filename))

            path_base.path.mkdir(mode=0o777, parents=True, exist_ok=True)
            path_zip.copy(_spath_dst=path_zip_dst.str)
            self._cd_target(_spath_base=path_base.str)
            self._decompress(_path_zip=path_zip_dst.str)
            path_zip_dst.unlink()
            self._cd_cwd()
        else:
            self._decompress(_path_zip=path_zip.str)

    def _cd_cwd(self) -> None:
        self.path_cwd.cd()

    def _cd_target(self, _spath_base: str) -> None:
        path_temp = path_.Path(_spath=_spath_base)
        if path_temp.path.is_dir() is True:
            path_target = path_temp
        else:
            path_target = path_temp.parent

        path_target.cd()

    def _check_validation(self, _spath_base: Optional[str], _targets: List[str]) -> None:
        if _spath_base is None:
            spath_base = ''
        else:
            spath_base = _spath_base
        for _idx, _target, in enumerate(_targets):
            path_target = path_.Path(_spath=os.path.join(spath_base, _target))
            if path_target.path.exists() is False:
                raise FileNotFoundError('{}'.format(path_target.str))

    def _compress_compact(self, _spath_zip: str, _targets: List[str], _spath_base: str, _level: int = zipfile.ZIP_DEFLATED):
        path_base = path_.Path(_spath=_spath_base)
        path_zip = path_.Path(_spath=_spath_zip)

        self._check_validation(_spath_base=_spath_base, _targets=_targets)
        self._cd_target(_spath_base=_spath_base)

        if isinstance(_targets, list):
            type_target = list_.check_element_type_list(_list=_targets, _type=str)

            if type_target is True:
                with zipfile.ZipFile(path_zip.str, 'w', _level) as f:
                    if self.pwd is not None:
                        f.setpassword(pwd=self.pwd)
                    for _idx, _target in enumerate(_targets):
                        path_target = path_.Path(_spath=os.path.join(path_base.str, _target))
                        f.write(_target)
                        for _idy, _path_target in enumerate(path_target.path.rglob(pattern='*')):
                            f.write(str(_path_target).replace(os.getcwd(), '.'))
                self._cd_cwd()
            else:
                raise ValueError('The type for elements of the _targets may be incorrect.')
        else:
            raise ValueError('The type of _targets should be list.')

    def _compress_full(self, _spath_zip: str, _targets: List[str], _level: int = zipfile.ZIP_DEFLATED):
        path_zip = path_.Path(_spath=_spath_zip)

        self._check_validation(_spath_base=None, _targets=_targets)

        if isinstance(_targets, list):
            type_target = list_.check_element_type_list(_list=_targets, _type=str)

            if type_target is True:
                with zipfile.ZipFile(path_zip.str, 'w', _level) as f:
                    if self.pwd is not None:
                        f.setpassword(pwd=self.pwd)
                    for _idx, _target in enumerate(_targets):
                        path_target = path_.Path(_spath=os.path.join(_target))
                        f.write(_target)
                        for _idy, _path_target in enumerate(path_target.path.rglob(pattern='*')):
                            f.write(str(_path_target))
            else:
                raise ValueError('The type for elements of the _targets may be incorrect.')
        else:
            raise ValueError('The type of _targets should be list.')

    def _decompress(self, _path_zip: str) -> None:
        zip_file = zipfile.ZipFile(_path_zip)
        if self.pwd is not None:
            zip_file.setpassword(pwd=self.pwd)
        zip_file.extractall()
        zip_file.close()