"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Nov. 2, 2020.

Title: vujade_videocv.py
Version: 0.1.2
Description: A module for video processing with computer vision.
"""


import os
import numpy as np
import math
import cv2
import ffmpeg
from vujade.utils.SceneChangeDetection import cy_scd


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
    def __init__(self, _path_video, _resolution=(1920, 1080), _fps=30.0, _qp_val=0, _pix_fmt='bgr24', _codec='libx264'):
        if _path_video is None:
            raise ValueError('The parameter, _path_video, should be assigned.')

        self.path_video = _path_video
        self.width = int(_resolution[0])
        self.height = int(_resolution[1])
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
    def __init__(self, _path_video, _resolution=(1920, 1080), _fps=30.0, _fourcc=cv2.VideoWriter_fourcc(*'MJPG')):
        if _path_video is None:
            raise Exception('The variable, _path_video, should be assigned.')

        self.path_video = _path_video
        self.width = int(_resolution[0])
        self.height = int(_resolution[1])
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


class SceneChangeDetectorFFmpeg:
    # ref.: https://rusty.today/posts/ffmpeg-scene-change-detector
    #
    # FFmpeg command:
    #    i)  ffmpeg -i _path_video -filter:v "select='gt(scene, 0.4)', showinfo" -f null - 2> ffout.log
    #    ii) grep showinfo ffout.log | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > ffout_scene_change_detection.log

    def __init__(self, _frame_sz=None, _threshold=0.4, _cython=True):
        if _frame_sz is None:
            raise ValueError('The argument should be tuple, not None.')

        self.threshold_val = _threshold
        self.cython = _cython
        self.frame_sz = _frame_sz
        self.width = _frame_sz[0]
        self.height = _frame_sz[1]
        self.nb_sad = 3 * self.height * self.width
        self.cnt_call = 0
        self.mafd_prev = None
        self.mafd_curr = None
        self.diff_curr = None
        self.scence_change_val = None
        self.res = None
        self.ndarr_frame_curr = None
        self.ndarr_frame_ref = None

    def get_frame_index(self, _path_video):
        res = []
        vid_src = VideoReaderCV(_path_video=_path_video)
        for idx in range(int(vid_src.num_frames)):
            ndarr_frame = cv2.resize(np.squeeze(vid_src.imread(_num_batch_frames=1, _trans=None)), dsize=self.frame_sz, interpolation=cv2.INTER_LINEAR)
            is_scene_change = self.run(_ndarr_frame=ndarr_frame, _be_float32=True)
            if is_scene_change is True:
                res.append(idx)

        return res

    def run(self, _ndarr_frame, _be_float32=True):
        if _ndarr_frame is None:
            raise ValueError('The given ndarr_frame should be assigned.')

        if _be_float32 is True:
            self.ndarr_frame_curr = _ndarr_frame.astype(np.float32)
        else:
            self.ndarr_frame_curr = _ndarr_frame

        self._get_scene_change_score()
        self._detect_scene_change()

        return self.res

    def _get_scene_change_score(self):
        if self.cnt_call == 0:
            pass
        elif self.cnt_call == 1:
            self._check_dimension()
            self._get_mafd()
        else:
            self._check_dimension()
            self._get_mafd()
            self._get_diff()
            self.scence_change_val = self._calculate_scene_change_value()

        self._update()

    def _detect_scene_change(self):
        if self.scence_change_val is None:
            self.res = None # Pending
        else:
            if self.threshold_val <= self.scence_change_val:
                self.res =True # Scene change
            else:
                self.res = False # No scene change

    def _check_dimension(self):
        if self.cython is True:
            cy_scd.check_dimension(_ndarr_1=self.ndarr_frame_curr, _ndarr_2=self.ndarr_frame_ref)
        else:
            if self.ndarr_frame_curr.shape != self.ndarr_frame_ref.shape:
                raise ValueError('The given both frames should have equal shape.')

    def _get_mafd(self):
        if self.cython is True:
            self.mafd_curr = cy_scd.mafd(_ndarr_1=self.ndarr_frame_curr, _ndarr_2=self.ndarr_frame_ref, _nb_sad=self.nb_sad)
        else:
            sad = self._get_sad()

            if self.nb_sad == 0:
                self.mafd_curr = 0.0
            else:
                self.mafd_curr = sad / self.nb_sad

    def _get_diff(self):
        if self.cython is True:
            self.diff_curr = cy_scd.diff(_val_1=self.mafd_curr, _val_2=self.mafd_prev)
        else:
            self.diff_curr = abs(self.mafd_curr - self.mafd_prev)

    def _calculate_scene_change_value(self):
        if self.cython is True:
            res = cy_scd.calculate_scene_change_value(_mafd=self.mafd_curr, _diff=self.diff_curr, _min=0.0, _max=1.0)
        else:
            res = self._clip(_val=min(self.mafd_curr, self.diff_curr) / 100.0, _min=0.0, _max=1.0)

        return res

    def _get_sad(self):
        return np.sum(np.fabs(self.ndarr_frame_curr - self.ndarr_frame_ref))

    def _clip(self, _val, _min=0.0, _max=1.0):
        if _val <= _min:
            _val = _min
        if _max <= _val:
            _val = _max

        return _val

    def _update(self):
        self.ndarr_frame_ref = self.ndarr_frame_curr
        self.mafd_prev = self.mafd_curr
        self.cnt_call += 1
