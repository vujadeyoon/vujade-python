"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_omegaconf.py
Description: A module for OmegaConf
"""


import omegaconf
from omegaconf import OmegaConf as _OmeagaConf
from typing import Optional
from vujade.vujade_debug import printf


class OmegaConf(object):
    @classmethod
    def load(cls, _spath_filename: str) -> omegaconf.dictconfig.DictConfig:
        cfg = _OmeagaConf.load(_spath_filename)
        cls._check(_cfg=cfg, _key_recursived=None)
        return cfg

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
            error_traced = str(printf(e, _is_print=False, _is_pause=False))
            raise UserWarning(error_traced)


if __name__ == '__main__':
    conf = OmegaConf.load('cfg.yaml')
