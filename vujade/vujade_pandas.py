"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_pandas.py
Description: A module for pandas
"""


import pandas as pd
from typing import Any, Optional, Union
from vujade.vujade_debug import printd


def get_matched_df(_df: pd.core.frame.DataFrame, _key: Union[int, str], _val: Any) -> Optional[pd.core.frame.DataFrame]:
    if _key in _df.keys():
        res = _df[_df[_key] == _val]
    else:
        printd('[WARNING] The given key (i.e. {}) is not existed in keys (i.e. {}).'.format(_key, _df.keys()), _is_pause=False)
        res = None
    return res
