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
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(_num_len_str))
