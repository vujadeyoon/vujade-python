"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: main_profiler.py
Description: A main python script to profile time, main memory and gpu memory.
"""


import time
import argparse
from vujade import vujade_utils as utils_
from vujade import vujade_resource as res_


parser = argparse.ArgumentParser(description='Main and GPU memory profiler')
parser.add_argument('--pid', type=int, required=True, help='Process ID (PID)')
parser.add_argument('--gpu_id', type=int, default=0, help='Graphics Processing Unit (GPU) ID')
parser.add_argument('--mem_main_warning', type=int, default=4 * 1024, help='Unit: MiB')
parser.add_argument('--mem_gpu_warning', type=int, default=4 * 1024, help='Unit: MiB')
parser.add_argument('--ratio_mem_fatal', type=float, default=0.9, help='Ratio for fatal memory usage')
parser.add_argument('--unit', type=int, default=3, help='Unit')
args = parser.parse_args()


if __name__ == '__main__':
    pid = args.pid
    gpu_id = args.gpu_id
    unit = args.unit
    mem_main_warning = args.mem_main_warning
    mem_gpu_warning = args.mem_gpu_warning
    ratio_mem_fatal = args.ratio_mem_fatal

    mem_main = res_.MainMemory(_pid=pid)
    mem_gpu = res_.GPUMemory(_pid=pid, _gpu_id=gpu_id)

    mem_main_fatal = ratio_mem_fatal * mem_main.get_mem_main_total()
    mem_gpu_fatal = ratio_mem_fatal * mem_gpu.get_mem_gpu_total()

    while True:
        dict_datetime_curr, _ = utils_.get_datetime()
        year = dict_datetime_curr['year']
        month = dict_datetime_curr['month']
        day = dict_datetime_curr['day']
        minute = dict_datetime_curr['minute']
        second = dict_datetime_curr['second']
        asctime = '{} {}:{}:{}'.format(year + month + day, hour, minute, second)

        mem_main_curr = mem_main.get_mem_main_proc()
        mem_gpu_curr = mem_gpu.get_mem_gpu_used()

        info = '[{:s}] [PID: {:d}, GPU ID: {:d}] Main memory: {:.2f} MiB, GPU memory: {:.2f} MiB.'.format(asctime, pid, gpu_id, mem_main_curr, mem_gpu_curr)

        bcolor = 'ENDC'
        if (mem_main_warning <= mem_main_curr) or (mem_gpu_warning <= mem_gpu_curr):
            bcolor='WARNING'
        if (mem_main_fatal <= mem_main_curr) or (mem_gpu_fatal <= mem_gpu_curr):
            bcolor='FATAL'

        utils_.print_color(_str=info, _bcolor=bcolor)

        time.sleep(unit)
