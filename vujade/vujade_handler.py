"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_handler.py
Description: A module for handler
"""


def signal_handler(_sig, _frame, _message: str = 'The commands, Ctrl+C, are entered.') -> None:
    # Usage:
    #     i)  signal.signal(signal.SIGINT, handler_.signal_handler)
    #     ii) signal.pause()
    global is_hdl
    is_hdl = False
    print(_message)
