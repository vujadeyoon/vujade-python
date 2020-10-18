"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Sep. 27, 2020.

Title: vujade_videocv.py
Version: 0.1.1
Description: A module for video processing with computer vision.
"""


import os
import numpy as np
import math
import cv2
import ffmpeg


class VideoReaderFFmpeg:
    def __init__(self, _path_video, _channel=3, _pix_fmt='bgr24'):
        self.path_video = _path_video
        video_info = self._get_info()
        self.height = video_info['height']
        self.width = video_info['width']
        self.channel = _channel
        self.fps = eval(video_info['avg_frame_rate'])
        self.time = eval(video_info['duration'])
        self.num_frames = math.ceil(self.fps * self.time)
        self.pix_fmt = _pix_fmt
        self.idx_frame_curr = -1
        self.num_frames_remain = self.num_frames

        self.cap = (
            ffmpeg
            .input(self.path_video)
            .output('pipe:', format='rawvideo', pix_fmt=self.pix_fmt)
            .run_async(pipe_stdout=True)
        )

    def imread(self, _num_batch_frames=1, _trans=(0, 3, 1, 2)):
        if self.num_frames_remain < _num_batch_frames:
            _num_batch_frames = self.num_frames_remain # equivalent: %=

        in_bytes = self.cap.stdout.read((self.width * self.height * self.channel) * _num_batch_frames)

        if not in_bytes:
            return None

        frames = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([-1, self.height, self.width, self.channel])
        )

        if _trans is not None:
            frames = frames.transpose(_trans)

        self.idx_frame_curr += _num_batch_frames
        self.num_frames_remain -= _num_batch_frames
        self._cal_eof()

        return frames

    def _cal_eof(self):
        self.is_eof = (self.num_frames_remain == 0)

    def _get_info(self):
        probe = ffmpeg.probe(self.path_video)
        return next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)


class VideoWriterFFmpeg:
    def __init__(self, _path_video, _resolution=(1080, 1920), _fps=30.0, _qp_val=0, _pix_fmt='bgr24', _codec='libx264'):
        if _path_video is None:
            raise ValueError('The parameter, _path_video, should be assigned.')

        self.path_video = _path_video
        self.height = int(_resolution[0])
        self.width = int(_resolution[1])
        self.fps = float(_fps)
        self.qp_val = _qp_val
        self.pix_fmt = _pix_fmt
        self.codec = _codec

        self.wri = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=self.pix_fmt, s='{}x{}'.format(self.width, self.height))
            .filter('fps', fps=self.fps, round='up')
            .output(self.path_video, pix_fmt='yuv420p', **{'c:v': self.codec}, **{'qscale:v': self.qp_val})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def imwrite(self, _list_img):
        for idx, img in enumerate(_list_img):
            self.wri.stdin.write(img)

    def close(self):
        self.wri.stdin.close()
        self.wri.wait()


class VideoReaderCV:
    def __init__(self, _path_video, _sec_start=None, _sec_end=None):
        if _path_video is None:
            raise Exception('The parameter, _path_video, should be assigned.')

        if (_sec_start is not None) and (isinstance(_sec_start, int) is False):
            raise Exception('The parameter, _sec_start, should be None or integer.')

        if (_sec_end is not None) and (isinstance(_sec_end, int) is False):
            raise Exception('The parameter, _sec_end, should be None or integer.')

        if (isinstance(_sec_start, int) is True) and (isinstance(_sec_end, int) is True) and (_sec_start >= _sec_end):
            raise Exception('The parameter _sec_start should be lower than the parameter _sec_end.')

        self.path_video = _path_video
        self.cap = self._open()
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.num_frames_ori = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.length_ori = int(self.num_frames_ori / self.fps)

        if (_sec_end is not None) and (self.length_ori <= _sec_end):
            _sec_end = None

        if (_sec_start is None) or ((_sec_start is not None) and (_sec_start < 0)):
            self.frame_start = 0
            self.sec_start = 0
        else:
            self.sec_start = _sec_start
            self.frame_start = int(self.sec_start * self.fps)

        if (_sec_end is None) or ((_sec_end is not None) and (_sec_end < 0)):
            self.frame_end = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            self.sec_end = int(self.frame_end / self.fps)
        else:
            self.sec_end = _sec_end
            self.frame_end = int(self.sec_end * self.fps)

        self.num_frames = self.frame_end - self.frame_start + 1
        self._set(_idx_frame=self.frame_start - 1)
        self.frame_timestamps = []
        self.is_eof = False

    def _is_open(self):
        return self.cap.isOpened()

    def _open(self):
        self.cap = cv2.VideoCapture(self.path_video)

        if self._is_open() is False:
            raise ValueError('The video capture is not opened.')

        return self.cap

    def _cal_eof(self):
        self.is_eof = (self.idx_frame_curr >= self.frame_end)

    def _set(self, _idx_frame):
        '''
        :param _idx_frame: Interval: [0, self.frame_end-1]
        '''

        if self.frame_end <= _idx_frame:
            raise ValueError('The parameter, _idx_frame, should be lower than self.frame_end.')

        self.cap.set(cv2.CAP_PROP_FRAME_COUNT, _idx_frame)
        self.idx_frame_curr = _idx_frame
        self._cal_eof()

    def _read(self):
        ret, frame = self.cap.read()
        self._timestamps()
        self.idx_frame_curr += 1
        self._cal_eof()

        return frame

    def _timestamps(self):
        self.frame_timestamps.append(self.cap.get(cv2.CAP_PROP_POS_MSEC))

    def imread(self, _num_batch_frames=1, _trans=(0, 3, 1, 2), _set_idx_frame=None):
        if _set_idx_frame is not None:
            self._set(_set_idx_frame)

        frames = None
        for idy in range(_num_batch_frames):
            if self.is_eof is True:
                break

            frame_src = np.expand_dims(self._read(), axis=0)

            if idy == 0:
                frames = frame_src
            else:
                frames = np.concatenate((frames, frame_src), axis=0)

        if _trans is not None:
            frames = frames.transpose(_trans)

        return frames

    def close(self):
        self.cap.release()


class VideoWriterCV:
    def __init__(self, _path_video, _resolution=(1080, 1920), _fps=30.0, _fourcc=cv2.VideoWriter_fourcc(*'MJPG')):
        if _path_video is None:
            raise Exception('The variable, _path_video, should be assigned.')

        self.path_video = _path_video
        self.height = int(_resolution[0])
        self.width = int(_resolution[1])
        self.fps = float(_fps)
        self.fourcc = _fourcc
        self.wri = self._open()

    def imwrite(self, _list_img):
        for idx, img in enumerate(_list_img):
            self.wri.write(image=img)

    def _open(self):
        return cv2.VideoWriter(self.path_video, self.fourcc, self.fps, (self.width, self.height))

    def close(self):
        self.wri.release()


def encode_vid2vid(_path_video_src, _path_video_dst):
    os.system('ffmpeg -i {} {}'.format(_path_video_src, _path_video_dst))
