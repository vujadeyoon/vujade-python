"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_network.py
Description: A module for network
"""


import socket


def get_ip_adrr() -> str:
    try:
        res = socket.gethostbyname(socket.gethostname())
        # res = [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
    except:
        res = 'unknown'

    return res
