"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_random.py
Description: A module for random
"""


import string
import random


def get_random_string(_num_len_str = 5) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=_num_len_str))
