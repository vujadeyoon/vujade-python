"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_argparse.py
Description: A module for argparse
"""


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
