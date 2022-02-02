"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_network.py
Description: A module for network
"""


import socket


def get_ip_addr_1() -> str:
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        res = st.getsockname()[0]
    except Exception:
        res = 'Unknown'
    finally:
        st.close()

    return res


def get_ip_adrr_2() -> str:
    try:
        res = [(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]
        # res = socket.gethostbyname(socket.gethostname())
    except:
        res = 'unknown'

    return res
