"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_argparse.py
Description: A module for argparse
"""


import argparse
import ast
from typing import Union


class ArgumentHandler(object):
    @staticmethod
    def str2any(_arg: str) -> list:
        if isinstance(_arg, (str, )):
            try:
                res = ast.literal_eval(_arg)
            except Exception as e:
                res = _arg
        else:
            res = _arg

        return res

    @staticmethod
    def str2bool(_v: Union[str, bool]) -> bool:
        # This function is equivalent to the built-in function, bool(strtobool()), in the distutils.util.
        if isinstance(_v, (bool, )):
            return _v
        if _v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif _v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


class Args(object):
    def __init__(self, _args: dict) -> None:
        super(Args, self).__init__()
        self.args = _args

    def __str__(self) -> str:
        return self.str

    def __repr__(self) -> str:
        return self.str

    @property
    def str(self) -> str:
        return str(self.args)


class Dict2Args(object):
    def __init__(self, _args: dict) -> None:
        super(Dict2Args, self).__init__()
        self.args_dict = _args

    def run(self) -> Args:
        args = Args(_args=self.args_dict)

        for _idx, (_key, _val) in enumerate(self.args_dict.items()):
            setattr(args, _key, _val)

        return args
