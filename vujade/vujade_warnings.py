"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_warnings.py
Description: A module for warnings.

Reference:
    i)   https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    ii)  https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=hgh73&logNo=220647435252
"""


import warnings
from typing import Iterable, Tuple, Type
from vujade.vujade_debug import printd


class Warnings(object):
    actions = ("error", "ignore", "always", "default", "module", "once")

    @staticmethod
    def warn(_message: str) -> None:
        warnings.warn(_message)

    @staticmethod
    def get_filters(_is_pause: bool = True) -> list:
        return warnings.filters

    @classmethod
    def filterwarnings_modules(cls, _modules: Iterable[Tuple[str, str, int]]) -> None:
        """
        Usage:
            modules_warnings = (('ignore', 'torch.nn.functional', 718), )
            warnings_.Warnings.filterwarnings_modules(_modules=modules_warnings)
        """
        for _idx, (_action, _module, _lineno) in enumerate(_modules):
            assert isinstance(_action, str)
            assert isinstance(_module, str)
            assert isinstance(_lineno, int)
            cls.filterwarnings(_action=_action, _module=_module, _lineno=_lineno)

    @classmethod
    def filterwarnings(cls, _action: str = 'ignore', _message: str = '', _category: Type[Warning] = Warning, _module: str = '', _lineno: int = 0, _append: bool = False) -> None:
        """
        Usage:
            i)  warnings_.Warnings.filterwarnings(_action='ignore', _message='Custom user warning message')
            ii) warnings_.Warnings.filterwarnings(_action='ignore', _module='torch.nn.functional', _lineno=718)
        """

        cls.__check_action(_action=_action)

        warnings.filterwarnings(
            action=_action,
            message='' if _message == '' else '.*{}.*'.format(_message),
            category=_category,
            module='' if _module == '' else '.*{}.*'.format(_module),
            lineno=_lineno,
            append=_append
        )

    @classmethod
    def ignore(cls) -> None:
        # Usage: warnings_.Warnings.ignore()
        cls.__simplefilter(_action='ignore')

    @classmethod
    def default(cls) -> None:
        # Usage: warnings_.Warnings.default()
        cls.__simplefilter(_action='default')

    @classmethod
    def __simplefilter(cls, _action: str = 'ignore', _category: Type[Warning] = Warning, _lineno: int = 0, _append: bool = False) -> None:
        cls.__check_action(_action=_action)

        warnings.simplefilter(
            action=_action,
            category=_category,
            lineno=_lineno,
            append=_append
        )

    @classmethod
    def __check_action(cls, _action: str):
        if _action not in cls.actions:
            raise ValueError('The _action, {} has not supported yet.'.format(_action))


if __name__=='__main__':
    modules_warnings = (('ignore', 'torch.nn.functional', 718), )
    Warnings.filterwarnings_modules(_modules=modules_warnings)
