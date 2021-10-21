"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_time.py
Description: A module for time
"""


import datetime
from pytz import timezone


def get_datetime(_timezone=timezone('Asia/Seoul')) -> dict:
    datetime_object = datetime.datetime.now(_timezone)
    year = '{:02d}'.format(datetime_object.year % 2000)
    month = '{:02d}'.format(datetime_object.month)
    day = '{:02d}'.format(datetime_object.day)
    hour = '{:02d}'.format(datetime_object.hour)
    minute = '{:02d}'.format(datetime_object.minute)
    second = '{:02d}'.format(datetime_object.second)
    microsecond = '{:06d}'.format(datetime_object.microsecond)

    dict_datetime_curr = {'year': year,
                          'month': month,
                          'day': day,
                          'hour': hour,
                          'minute': minute,
                          'second': second,
                          'microsecond': microsecond
                          }
    datetime_curr_default = '{}.{}'.format(year + month + day + hour + minute + second, microsecond)
    datetime_curr_readable = '{}-{}-{} {}:{}:{}.{}'.format(year, month, day, hour, minute, second, microsecond)

    res = {'dict': dict_datetime_curr,
           'default': datetime_curr_default,
           'readable': datetime_curr_readable
           }

    return res


def hmsmss2smss(_hmsmss: str) -> str:
    hour, minute, sec_mss = _hmsmss.split(':')
    sec, mss = sec_mss.split('.')
    return '{}.{}'.format((3600 * int(hour)) + (60 * int(minute)) + int(sec), mss)


def s2hmsmss(_s: str) -> str:
    hour = int(int(_s) / 3600)
    minute = int(int(int(_s) % 3600) / 60)
    sec = int(int(int(_s) % 3600) % 60)
    mss = 0
    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(hour, minute, sec, mss)
