"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_builtins.py
Description: A module for builtins
"""


import builtins
from typing import List, Tuple, Union


class Builtins(object):
    def __init__(self, _name_builtins: Union[List[str], Tuple[str]]) -> None:
        super(Builtins, self).__init__()
        self.name_builtins = _name_builtins
        self.object_builtins = self._backup_object_builtins()

    def _backup_object_builtins(self) -> dict:
        res = dict()
        for _idx, _name_builtin in enumerate(self.name_builtins):
            res[_name_builtin] = getattr(builtins, _name_builtin)

        return res

    def update_builtins(self, _objects: Union[List[object], Tuple[object]]) -> None:
        if len(self.name_builtins) != len(_objects):
            raise ValueError('The length of the given _objects should be equal to that of the self.name_builtins.')

        for _idx, (_name_builtin, _object) in enumerate(zip(self.name_builtins, _objects)):
            setattr(builtins, _name_builtin, _object)

    def restore_builtins(self) -> None:
        for _idx, _name_builtin in enumerate(self.name_builtins):
            if _name_builtin in self.object_builtins.keys():
                setattr(builtins, _name_builtin, self.object_builtins[_name_builtin])
            else:
                raise ValueError('The given _name_builtin, {} is not in the self.object_builtins.'.format(_name_builtin))
