"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: main_profiler.py
Description: A main python script to profile time, main memory and GPU memory.
"""


import os
import sys
import time
import argparse
try:
    from vujade import vujade_utils as utils_
    from vujade import vujade_time as time_
    from vujade import vujade_resource as res_
except Exception as e:
    sys.path.append(os.path.join(os.getcwd()))
    from vujade import vujade_utils as utils_
    from vujade import vujade_time as time_
    from vujade import vujade_resource as res_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main and GPU memory profiler')
    parser.add_argument('--pid', type=int, required=True, help='Process ID (PID)')
    parser.add_argument('--gpu_id', type=int, default=0, help='Graphics Processing Unit (GPU) ID')
    parser.add_argument('--mem_main_warning', type=int, default=4 * 1024, help='Unit: MiB')
    parser.add_argument('--mem_gpu_warning', type=int, default=4 * 1024, help='Unit: MiB')
    parser.add_argument('--ratio_mem_fatal', type=float, default=0.9, help='Ratio for fatal memory usage')
    parser.add_argument('--unit', type=int, default=3, help='Unit')
    args = parser.parse_args()

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
        asctime = time_.get_datetime()['readable']

        mem_main_curr = mem_main.get_mem_main_proc()
        mem_gpu_curr = mem_gpu.get_mem_gpu_used()

        info = '[{:s}] [PID: {:d}, GPU ID: {:d}] Main memory: {:.2f} MiB, GPU memory: {:.2f} MiB.'.format(asctime, pid, gpu_id, mem_main_curr, mem_gpu_curr)

        bcolor = 'ENDC'
        if (mem_main_warning <= mem_main_curr) or (mem_gpu_warning <= mem_gpu_curr):
            bcolor='WARNING'
        if (mem_main_fatal <= mem_main_curr) or (mem_gpu_fatal <= mem_gpu_curr):
            bcolor='FATAL'

        utils_.print_color(_str=info, _color=bcolor)

        time.sleep(unit)
