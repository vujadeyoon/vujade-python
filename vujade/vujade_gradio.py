"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_gradio.py
Description: A module for gradio
"""


import abc
import gradio as gr
import numpy as np
import cv2
from typing import Set
from vujade import vujade_path as path_
from vujade.vujade_debug import printd, pprintd


class _WebDemoGradio(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def algorithms(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def launch(self) -> None:
        pass


class WebDemoFromImageVideo(_WebDemoGradio):
    def __init__(self) -> None:
        super(WebDemoFromImageVideo, self).__init__()
        path_dir_src = gr.File(file_count='single', file_types=['.mp4'], label='Upload a directory for images or a video')
        self.demo = gr.Interface(fn=self.algorithms, inputs=gr.Video(), outputs=[gr.Video()], title='Image Flipper', cache_examples=True)

    def get_img_extension(self) -> Set[str]:
        return {'.bmp', '.jpg', 'jpeg', '.png'}

    def get_vid_extension(self) -> Set[str]:
        return {'.avi', '.mp4', '.yuv'}


    def _is_check_files(self, _path_dir_src: list) -> None:
        res = set()
        for _path_file_src in _path_dir_src:
            res.add(_path_file_src.ext)

        if (res.issubset(self.get_img_extension())) and (res.issubset(self.get_vid_extension())):
            raise ValueError('The uploaded files should consist of only images or videos.')

    def algorithms(self, video):
        a = np.asarray(video)
        printd(video, type(video), type(a), a.shape, a.dtype)
        return video

    def algorithms_1(self, path_dir_src: list) -> tuple:
        path_dir_src = [path_.Path(_path_dir_src) for _path_dir_src in sorted(path_dir_src)]
        self._is_check_files(_path_dir_src=path_dir_src)

        for _idx, _path_file_src in enumerate(path_dir_src):
            res = cv2.imread(_path_file_src.str)

        return res, '{}'.format(_idx)

    def launch(self) -> None:
        self.demo.launch()


if __name__=='__main__':
    # Usage: python3 ./vujade/vujade_gradio.py --path_dir_img_src /DATA/sjyoon1671/Develop/GitHub/vujade-python/asset/video/videoplayback
    demo = WebDemoFromImageVideo()
    demo.launch()
