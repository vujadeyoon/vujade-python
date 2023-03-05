"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: __init__.py
Description: A method-wrapper for the package, vujade.
"""


__date__ = '230305'
__version__ = '0.6.0'
__all__ = [
    'main_encdec',
    'main_img2vid',
    'main_profiler',
    'main_profiler_gpu_android',
    'vujade_argparse',
    'vujade_aws',
    'vujade_bytes',
    'vujade_compression',
    'vujade_cryptography',
    'vujade_csv',
    'vujade_datastructure',
    'vujade_debug',
    'vujade_dnn',
    'vujade_download',
    'vujade_erase',
    'vujade_flops_counter',
    'vujade_google',
    'vujade_handler',
    'vujade_imgcv',
    'vujade_json',
    'vujade_kube',
    'vujade_list',
    'vujade_logger',
    'vujade_loss',
    'vujade_lr_scheduler',
    'vujade_metric',
    'vujade_multiprocess',
    'vujade_multithread',
    'vujade_network',
    'vujade_nms',
    'vujade_openai',
    'vujade_opencv',
    'vujade_path',
    'vujade_profiler',
    'vujade_random',
    'vujade_resource',
    'vujade_segmentation',
    'vujade_slack',
    'vujade_str',
    'vujade_tensorboard',
    'vujade_text',
    'vujade_time',
    'vujade_transforms',
    'vujade_utils',
    'vujade_videocv',
    'vujade_warnings',
    'vujade_xlsx',
    'vujade_yaml',
    ]


import os
import logging
from pathlib import Path
from pytz import timezone
from typing import Optional


class VujadeLog(object):
    @staticmethod
    def get_level_verbose() -> int:
        try:
            res = int(os.environ.get('LEVEL_VERBOSE', ''))
        except ValueError as e:
            res = 3

        return res

    @staticmethod
    def get_spath_log(_timezone=timezone('Asia/Seoul')) -> Optional[str]:
        path_log = Path(os.environ.get('PATH_LOG', ''))

        if path_log.suffix in {'.log', '.txt'}:
            res = '{}'.format(path_log)
        else:
            res = None

        return res

    @staticmethod
    def get_log_level() -> int:
        log_level = os.environ.get('LOG_LEVEL', 'DEBUG')

        try:
            res = getattr(logging, log_level)
        except AttributeError as e:
            res = getattr(logging, 'DEBUG')

        return res


env_var = {
    'verbose': {
        'level': VujadeLog.get_level_verbose(),
    },
    'log': {
        'path': VujadeLog.get_spath_log(),
        'level': VujadeLog.get_log_level(),
        'is_traceback_print_stack': False,
        'builtins': {'print', 'input'},
    }
}
