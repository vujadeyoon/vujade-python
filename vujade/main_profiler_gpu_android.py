"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: main_profiler_gpu_android.py
Description: A main python script to profile GPU memory on Android device.
"""


import argparse
import time
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_gpu_min', type=int, default=30000, help='Maximum GPU temperature')
    parser.add_argument('--temp_gpu_max', type=int, default=50000, help='Minimum GPU temperature')
    parser.add_argument('--temp_gpu_criteria', type=int, default=40, help='GPU temperature criteria')
    parser.add_argument('--time_unit', type=int, default=1, help='Time unit')
    parser.add_argument('--thermal_zone', type=int, default=0, help='Thermal zone interval: [1, 99).')
    args = parser.parse_args()

    cmd = "adb shell 'cat /sys/class/thermal/thermal_zone0/temp'"

    while True:
        is_success, temp_gpu = utils_.SystemCommand.run(_command=cmd, _is_daemon=False, _is_subprocess=True)

        if is_success is True:
            temp_gpu = int(temp_gpu.rstrip().decode('utf-8'))
            ratio_usage_gpu = (temp_gpu - args.temp_gpu_min) / (args.temp_gpu_max - args.temp_gpu_min)
            if ratio_usage_gpu < 0.0:
                ratio_usage_gpu = 0.0
            elif 1.0 < ratio_usage_gpu:
                ratio_usage_gpu = 1.0
            else:
                pass
            percentage_usage_gpu = 100 * ratio_usage_gpu
            str_info = '[PID: {}] GPU temp.: {}; GPU usage: {:.2f}%.'.format(utils_.getpid(), temp_gpu, percentage_usage_gpu)
            if (args.temp_gpu_criteria < percentage_usage_gpu):
                printd(str_info + ' The GPU may be used.', _color='WARNING', _is_pause=False)
            else:
                printd(str_info, _is_pause=False)
            time.sleep(args.time_unit)
