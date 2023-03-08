"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_omegaconf.py
Description: A module for OmegaConf
"""


import omegaconf
from omegaconf import OmegaConf as _OmeagaConf
from typing import Any, Optional, Union
from vujade.vujade_debug import printd


class OmegaConf(object):
    @classmethod
    def load(cls, _spath_filename: str, _is_interpolation: bool = False) -> Union[dict, omegaconf.dictconfig.DictConfig]:
        cfg = _OmeagaConf.load(_spath_filename)
        cls._check(_cfg=cfg, _key_recursived=None)

        if _is_interpolation is True:
            res = dict()
            OmegaConf.cfg2dict(_res=res, _cfg=cfg, _key_recursived=None)
        else:
            res = cfg

        return res

    @staticmethod
    def cfg2dict(_res: dict, _cfg: omegaconf.dictconfig.DictConfig, _key_recursived: Optional[str] = None) -> None:
        for _idx, (_key, _val) in enumerate(_cfg.items()):
            if _key_recursived is None:
                key_recursived = _key
            else:
                key_recursived = '{}.{}'.format(_key_recursived, _key)

            if isinstance(_val, omegaconf.dictconfig.DictConfig) is True:
                OmegaConf.cfg2dict(_res=_res, _cfg=_val, _key_recursived=key_recursived)
            else:
                OmegaConf._update_dict(_key=key_recursived, _val=_val, _dict=_res)

    @staticmethod
    def _check(_cfg: omegaconf.dictconfig.DictConfig, _key_recursived: Optional[str] = None) -> None:
        try:
            for _idx, (_key, _val) in enumerate(_cfg.items()):
                if _key_recursived is None:
                    key_recursived = _key
                else:
                    key_recursived = '{}.{}'.format(_key_recursived, _key)

                if isinstance(_val, omegaconf.dictconfig.DictConfig) is True:
                    OmegaConf._check(_cfg=_val, _key_recursived=key_recursived)
        except Exception as e:
            error_traced = str(printd(e, _is_print=False, _is_pause=False))
            raise UserWarning(error_traced)

    @staticmethod
    def _update_dict(_key: str, _val: Any, _dict: dict) -> None:
        keys = _key.split('.')
        dict_recursived = _dict

        for _idx, _key in enumerate(keys):
            if _key == keys[-1]:
                if isinstance(_val, omegaconf.listconfig.ListConfig):
                    _val = [_ele for _ele in _val]
                dict_recursived.update({_key: _val})
            elif _key not in dict_recursived.keys():
                dict_recursived.update({_key: dict()})
            else:
                pass
            dict_recursived = dict_recursived[_key]


if __name__ == '__main__':
    conf = OmegaConf.load(_spath_filename='cfg.yaml', _is_interpolation=True)
