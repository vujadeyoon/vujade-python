"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_videocv.py
Description: A module for video processing with computer vision.
"""


import os
import numpy as np
import math
import cv2
import ffmpeg
import shlex
import subprocess
import json
from typing import Optional, Set
from vujade import vujade_utils as utils_
from vujade import vujade_list as list_
from vujade.utils.SceneChangeDetection.InteractiveProcessing import scd as scd_ip_
from vujade.utils.SceneChangeDetection.BatchProcessing import scd as scd_bp_


def timestamp2smss(_timestamp: list) -> list:
    return list(map(lambda x: math.floor(x) / (10 ** 3), _timestamp))


def get_vid_extension() -> Set[str]:
    return {'.avi', '.mp4', '.yuv'}


def encode_vid2vid(_spath_video_src: str, _spath_video_dst: str) -> bool:
    cmd = 'ffmpeg -i {} {}'.format(_spath_video_src, _spath_video_dst)
    return utils_.SystemCommand.run(_command=cmd, _is_daemon=False)


class VideoReaderFFmpeg(object):
    def __init__(self, _spath_video: str, _channel: int = 3, _pix_fmt: str = 'bgr24'):
        super(VideoReaderFFmpeg, self).__init__()
        self.spath_video = _spath_video
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
            .input(self.spath_video)
            .output('pipe:', format='rawvideo', pix_fmt=self.pix_fmt)
            .run_async(pipe_stdout=True)
        )

    def imread(self, _num_batch_frames: int = 1, _trans: tuple = (0, 3, 1, 2)) -> np.ndarray:
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

    def _cal_eof(self) -> None:
        self.is_eof = (self.num_frames_remain == 0)

    def _get_info(self) -> dict:
        probe = ffmpeg.probe(self.spath_video)
        return next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)


class VideoWriterFFmpeg(object):
    def __init__(self, _spath_video: str, _size: tuple = (1080, 1920), _fps: float = 30.0, _qp_val: int = 0, _pix_fmt: str = 'bgr24', _codec: str = 'libx264'):
        super(VideoWriterFFmpeg, self).__init__()
        if _spath_video is None:
            raise ValueError('The parameter, _spath_video, should be assigned.')

        self.spath_video = _spath_video
        self.height = int(_size[0])
        self.width = int(_size[1])
        self.fps = float(_fps)
        self.qp_val = _qp_val
        self.pix_fmt = _pix_fmt
        self.codec = _codec

        self.wri = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt=self.pix_fmt, s='{}x{}'.format(self.width, self.height))
            .filter('fps', fps=self.fps, round='up')
            .output(self.spath_video, pix_fmt='yuv420p', **{'c:v': self.codec}, **{'qscale:v': self.qp_val})
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def imwrite(self, _list_img: list) -> None:
        for idx, img in enumerate(_list_img):
            self.wri.stdin.write(img)

    def close(self) -> None:
        self.wri.stdin.close()
        self.wri.wait()


class VideoReaderCV(object):
    def __init__(self, _spath_video: str, _sec_start: int = None, _sec_end: int = None) -> None:
        super(VideoReaderCV, self).__init__()
        if _spath_video is None:
            raise ValueError('The parameter, _spath_video, should be assigned.')

        if (_sec_start is not None) and (isinstance(_sec_start, int) is False):
            raise ValueError('The parameter, _sec_start, should be None or integer.')

        if (_sec_end is not None) and (isinstance(_sec_end, int) is False):
            raise ValueError('The parameter, _sec_end, should be None or integer.')

        if (isinstance(_sec_start, int) is True) and (isinstance(_sec_end, int) is True) and (_sec_start >= _sec_end):
            raise ValueError('The parameter _sec_start should be lower than the parameter _sec_end.')

        self.spath_video = _spath_video
        self._open()
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.channel = 3
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.num_frames_ori = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.length_ori = int(self.num_frames_ori / self.fps)
        self.orientation = self._get_orientation()
        self.frame_types = self._get_frame_types()

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
        self.is_random_access = self._check_random_access(_idx_frame=self.num_frames)
        self.idx_frame_curr = (self.frame_start - 1)
        self.num_frames_remain = self.frame_end - self.frame_start + 1
        self.frame_timestamps = []
        self.is_eof = False
        self._set(_idx_frame=self.frame_start)

    def _is_open(self) -> bool:
        return self.cap.isOpened()

    def _open(self) -> None:
        self.cap = cv2.VideoCapture(self.spath_video)

        if self._is_open() is False:
            raise ValueError('The video capture is not opened.')

    def _cal_eof(self) -> None:
        self.is_eof = (self.frame_end <= self.idx_frame_curr)

    def _set(self, _idx_frame: int) -> None:
        '''
        :param _idx_frame: Interval: [0, self.frame_end-1]
        '''

        if self.frame_end < _idx_frame:
            raise ValueError('The parameter, _idx_frame, should be lower than self.frame_end.')

        if self.idx_frame_curr <= _idx_frame:
            for idx in range((_idx_frame - self.idx_frame_curr) - 1):
                if self.is_eof is True:
                    break
                self._read(_is_record_timestamp=False)
        else:
            self._set_idx_frame(_idx_frame=_idx_frame)
            self.idx_frame_curr = (_idx_frame - 1)
            self.num_frames_remain = self._update_num_frames_reamin()
            self._cal_eof()


    def _read(self, _is_record_timestamp: bool = True) -> np.ndarray:
        ret, frame = self.cap.read()
        if _is_record_timestamp is True:
            self._timestamps()
        self.idx_frame_curr += 1
        self.num_frames_remain = self._update_num_frames_reamin()
        self._cal_eof()

        return frame

    def _update_num_frames_reamin(self) -> int:
        return self.frame_end - self.idx_frame_curr

    def _timestamps(self) -> None:
        self.frame_timestamps.append(self.cap.get(cv2.CAP_PROP_POS_MSEC))

    def _get_orientation(self) -> str:
        """
        Function to get the rotation of the input video file.
        Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
        stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

        Returns a rotation None, 90, 180 or 270
        """
        command = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 '{}'".format(self.spath_video)
        ffprobe_output = utils_.SystemCommand.check_output(_command=command, _split=True, _shell=False, _decode=True)

        if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
            ffprobe_output = json.loads(ffprobe_output)
            rotation = ffprobe_output
        else:
            rotation = 0

        if (rotation == 0) or (rotation == 180):
            orientation = 'horizontal'
        else:  # (rotation == 90) or (rotation == 270):
            orientation = 'vertical'

        return orientation

    def _get_frame_types(self) -> dict:
        command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1 {}'.format(self.spath_video)
        frame_types = utils_.SystemCommand.check_output(_command=command, _split=True, _shell=False, _decode=True).replace('pict_type=', '').split()

        res = {
            'I': list_.find_indices(_list=frame_types, _mached_element='I'),
            'P': list_.find_indices(_list=frame_types, _mached_element='P'),
            'B': list_.find_indices(_list=frame_types, _mached_element='B')
        }

        return res

    def _get_idx_frame(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def _set_idx_frame(self, _idx_frame: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, _idx_frame)

    def _check_random_access(self, _idx_frame: Optional[int]) -> bool:
        if _idx_frame is None:
            _idx_frame = self.num_frames

        idx_frame_curr = self._get_idx_frame()
        self._set_idx_frame(_idx_frame=_idx_frame)
        res = (self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.num_frames)
        self._set_idx_frame(_idx_frame=idx_frame_curr)

        return res

    def imread(self, _num_batch_frames: int = 1, _trans: tuple = None, _set_idx_frame: int = None, _dsize: tuple = None, _color_code: int = None, _interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        if (_dsize is None) or (_dsize == (self.width, self.height)):
            is_resize = False
            width = self.width
            height = self.height
        else:
            is_resize = True
            width = _dsize[0]
            height = _dsize[1]

        if _color_code is None:
            channel = self.channel
        elif _color_code == cv2.COLOR_BGR2GRAY:
            channel = 1
        else:
            raise NotImplementedError

        if (_set_idx_frame is not None) and (0 <= _set_idx_frame):
            self._set(_set_idx_frame)

        if _num_batch_frames <= self.num_frames_remain:
            frames = np.zeros(shape=(_num_batch_frames, height, width, channel), dtype=np.uint8)
            num_batch_frames = _num_batch_frames
        else:
            frames = np.zeros(shape=(self.num_frames_remain, height, width, channel), dtype=np.uint8)
            num_batch_frames = self.num_frames_remain

        for idx in range(num_batch_frames):
            if self.is_eof is True:
                break

            if is_resize is False:
                temp = self._read(_is_record_timestamp=True)
            else:
                temp = cv2.resize(src=self._read(_is_record_timestamp=True), dsize=(width, height), interpolation=_interpolation)

            if _color_code is None:
                frames[idx, :, :, :] = temp
            elif channel == 1:
                frames[idx, :, :, 0] = cv2.cvtColor(temp, code=_color_code)
            else:
                raise NotImplementedError

        if _trans is not None:
            frames = frames.transpose(_trans)

        return frames

    def get_timestamp(self) -> list:
        while (self.is_eof is False):
            self.imread(_num_batch_frames=1, _trans=None)

        return self.frame_timestamps # smss.mus

    def close(self) -> None:
        self.cap.release()


class VideoWriterCV(object):
    def __init__(self, _spath_video: str, _size: tuple = (1080, 1920), _fps: float = 30.0, _fourcc: int = cv2.VideoWriter_fourcc(*'MJPG')):
        super(VideoWriterCV, self).__init__()
        if _spath_video is None:
            raise ValueError('The variable, _spath_video, should be assigned.')

        self.spath_video = _spath_video
        self.height = int(_size[0])
        self.width = int(_size[1])
        self.fps = float(_fps)
        self.fourcc = _fourcc
        self.wri = self._open()

    def imwrite(self, _list_img: list) -> None:
        for idx, img in enumerate(_list_img):
            self.wri.write(image=img)

    def _open(self) -> cv2.VideoWriter:
        return cv2.VideoWriter(self.spath_video, self.fourcc, self.fps, (self.width, self.height))

    def close(self) -> None:
        self.wri.release()


class SceneChangeDetectorFFmpeg(object):
    """This class is intended to detect scene change for the given video.
    The reference is as follows: https://rusty.today/posts/ffmpeg-scene-change-detector.
    The corresponding FFmpeg is as below.
        i)  ffmpeg -i _spath_video -filter:v "select='gt(scene, 0.4)', showinfo" -f null - 2> ffout.log
        ii) grep showinfo ffout.log | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > ffout_scene_change_detection.log
    """
    def __init__(self, _threshold: float = 0.4):
        super(SceneChangeDetectorFFmpeg, self).__init__()
        self.threshold = _threshold
        self.offset = 9

    def get_frame_index(self, _spath_video: str) -> list:
        vid_src = VideoReaderCV(_spath_video=_spath_video)
        vid_src_timestamp = self._convert(_list=vid_src.get_timestamp(), _unit=1000, _decimals=4)

        command = self._get_command(_spath_video=_spath_video)
        str_stdout, str_stderr = utils_.get_stdout_stderr(_command=command)

        idx_start = utils_.find_substr(_str_src=str_stderr.decode('utf-8'), _str_sub='pts_time:')
        idx_end = utils_.find_substr(_str_src=str_stderr.decode('utf-8'), _str_sub=' pos:')

        scd_timestamp = []
        for idx, (_idx_start, _idx_end) in enumerate(zip(idx_start, idx_end)):
            scd_timestamp.append(float(str_stderr[_idx_start + self.offset:_idx_end]))

        res = list_.list_matching_idx(_list_1=self._convert(_list=scd_timestamp, _unit=1.0, _decimals=4), _list_2=vid_src_timestamp)

        return res

    def _get_command(self, _spath_video: str) -> str:
        return 'ffmpeg -i {} -filter:v \"select=\'gt(scene, {})\', showinfo\" -f null pipe:1'.format(_spath_video, self.threshold)

    def _convert(self, _list, _unit=1.0, _decimals=4):
        return list(np.round(np.array(_list) / _unit, _decimals))


class SceneChangeDetectorFFmpegInteractiveProcessing(object):
    """This class is intended to detect scene change for the given video.
    The reference is as follows: https://rusty.today/posts/ffmpeg-scene-change-detector.
    The corresponding FFmpeg is as below.
        i)  ffmpeg -i _spath_video -filter:v "select='gt(scene, 0.4)', showinfo" -f null - 2> ffout.log
        ii) grep showinfo ffout.log | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > ffout_scene_change_detection.log
    """
    def __init__(self, _dsize: tuple = None, _threshold: float = 0.4, _is_cython: bool = True):
        """
        :param tuple _dsize: An image size for computation
        :param float _threshold: A thershold value to determine wheter the scene change occurs
        :param bool _is_cython: A boolean variable to decide whether to use cython
        """
        super(SceneChangeDetectorFFmpegInteractiveProcessing, self).__init__()
        if _dsize is None:
            raise ValueError('The argument should be tuple, not None.')

        self.dsize = _dsize
        self.threshold = _threshold
        self.is_cython = _is_cython
        self.width = _dsize[0]
        self.height = _dsize[1]
        self.nb_sad = 3 * self.height * self.width
        self.cnt_call = 0
        self.mafd_prev = None
        self.mafd_curr = None
        self.diff_curr = None
        self.scence_change_val = None
        self.res = None
        self.ndarr_frame_curr = None
        self.ndarr_frame_ref = None

    def get_frame_index(self, _spath_video: str) -> list:
        """
        :param str _spath_video: A path for the given video file
        :returns: A list containing the frame index information where the scene change occurs
        """
        res = []
        vid_src = VideoReaderCV(_spath_video=_spath_video)
        for idx in range(int(vid_src.num_frames)):
            ndarr_frame = np.squeeze(vid_src.imread(_num_batch_frames=1, _trans=None, _dsize=self.dsize))
            is_scene_change = self.run(_ndarr_frame=ndarr_frame, _be_float32=True)
            if is_scene_change is True:
                res.append(idx)
        return res

    def run(self, _ndarr_frame: np.ndarray, _be_float32: bool = True) -> bool:
        if _ndarr_frame is None:
            raise ValueError('The given ndarr_frame should be assigned.')

        if _be_float32 is True:
            self.ndarr_frame_curr = _ndarr_frame.astype(np.float32)
        else:
            self.ndarr_frame_curr = _ndarr_frame

        self._get_scene_change_score()
        self._detect_scene_change()

        return self.res

    def _get_scene_change_score(self) -> None:
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

    def _detect_scene_change(self) -> None:
        if self.scence_change_val is None:
            self.res = None # Pending
        else:
            if self.threshold <= self.scence_change_val:
                self.res = True # Scene change
            else:
                self.res = False # No scene change

    def _check_dimension(self) -> None:
        if self.is_cython is True:
            scd_ip_.check_dimension(_ndarr_1=self.ndarr_frame_curr, _ndarr_2=self.ndarr_frame_ref)
        else:
            if self.ndarr_frame_curr.shape != self.ndarr_frame_ref.shape:
                raise ValueError('The given both frames should have equal shape.')

    def _get_mafd(self) -> None:
        if self.is_cython is True:
            self.mafd_curr = scd_ip_.mafd(_ndarr_1=self.ndarr_frame_curr, _ndarr_2=self.ndarr_frame_ref, _nb_sad=self.nb_sad)
        else:
            sad = self._get_sad()

            if self.nb_sad == 0:
                self.mafd_curr = 0.0
            else:
                self.mafd_curr = sad / self.nb_sad

    def _get_diff(self) -> None:
        if self.is_cython is True:
            self.diff_curr = scd_ip_.diff(_val_1=self.mafd_curr, _val_2=self.mafd_prev)
        else:
            self.diff_curr = abs(self.mafd_curr - self.mafd_prev)

    def _calculate_scene_change_value(self) -> float:
        if self.is_cython is True:
            res = scd_ip_.calculate_scene_change_value(_mafd=self.mafd_curr, _diff=self.diff_curr, _min=0.0, _max=1.0)
        else:
            res = self._clip(_val=min(self.mafd_curr, self.diff_curr) / 100.0, _min=0.0, _max=1.0)

        return res

    def _get_sad(self) -> np.ndarray:
        return np.sum(np.fabs(self.ndarr_frame_curr - self.ndarr_frame_ref))

    def _clip(self, _val, _min=0.0, _max=1.0) -> float:
        if _val <= _min:
            _val = _min
        if _max <= _val:
            _val = _max

        return _val

    def _update(self) -> None:
        self.ndarr_frame_ref = self.ndarr_frame_curr
        self.mafd_prev = self.mafd_curr
        self.cnt_call += 1


class SceneChangeDetectorFFmpegBatchProcessing(object):
    """This class is intended to detect scene change for the given video.
    The reference is as follows: https://rusty.today/posts/ffmpeg-scene-change-detector.
    The corresponding FFmpeg is as below.
        i)  ffmpeg -i _spath_video -filter:v "select='gt(scene, 0.4)', showinfo" -f null - 2> ffout.log
        ii) grep showinfo ffout.log | grep pts_time:[0-9.]* -o | grep [0-9.]* -o > ffout_scene_change_detection.log
    """
    def __init__(self, _dsize: tuple = None, _threshold: float = 0.4, _is_gray: bool = True, _unit_computation: int = 1800, _is_cython: bool = True):
        """
        :param tuple _dsize: An image size for computation
        :param float _threshold: A thershold value to determine wheter the scene change occurs
        :param bool _is_gray: A boolean variable to decide whether to be applied on grayscale
        :param bool _unit_computation: A computation unit
        :param bool _is_cython: A boolean variable to decide whether to use cython
        """
        super(SceneChangeDetectorFFmpegBatchProcessing, self).__init__()
        if _dsize is None:
            raise ValueError('The argument should be tuple, not None.')

        self.dsize = _dsize
        self.threshold = _threshold
        self.is_gray = _is_gray
        self.unit_computation = _unit_computation
        self.is_cython = _is_cython
        self.width = _dsize[0]
        self.height = _dsize[1]
        if self.is_gray is True:
            self.channel = 1
            self.color_code = cv2.COLOR_BGR2GRAY
        else:
            self.channel = 3
            self.color_code = None
        self.nb_sad = self.channel * self.height * self.width

        if self.nb_sad == 0:
            raise ValueError('The self.nb_sad should be positive.')

    def get_frame_index(self, _spath_video: str) -> list:
        """
        :param str _spath_video: A path for the given video file
        :returns: A list containing the frame index information where the scene change occurs
        """
        res = []
        vid_src = VideoReaderCV(_spath_video=_spath_video)
        while (vid_src.is_eof is False):
            offset = (vid_src.idx_frame_curr - 1)
            if offset < 0:
                offset = 0

            ndarr_frames = vid_src.imread(_num_batch_frames=self.unit_computation, _trans=(0, 3, 1, 2),
                                          _set_idx_frame=(vid_src.idx_frame_curr - 1),
                                          _dsize=(self.width, self.height),
                                          _color_code=self.color_code)

            mafd = self._get_mafd(_ndarr_frames=ndarr_frames)
            diff = self._get_diff(_mafd=mafd)
            scene_change_val = self._calculate_scene_change_value(_mafd=mafd, _diff=diff)
            idx_sc = self._get_idx_sc(_scene_change_val=scene_change_val, _threshold=self.threshold, _offset=offset)
            res.extend(list(idx_sc))

        return res

    def _get_mafd(self, _ndarr_frames) -> np.ndarray:
        if self.is_cython is True:
            res = scd_bp_.mafd(_ndarr_1=_ndarr_frames[1:, :, :, :].astype(np.int16), _ndarr_2=_ndarr_frames[:-1, :, :, :].astype(np.int16), _nb_sad=self.nb_sad)
        else:
            res = (np.sum(np.abs(_ndarr_frames[1:, :, :, :].astype(np.int16) - _ndarr_frames[:-1, :, :, :].astype(np.int16)), axis=(1, 2, 3))) / self.nb_sad

        return res

    def _get_diff(self, _mafd) -> np.ndarray:
        if self.is_cython is True:
            res = scd_bp_.diff(_mafd_1=_mafd[1:], _mafd_2=_mafd[:-1])
        else:
            res = np.abs(_mafd[1:] - _mafd[:-1])

        return res

    def _calculate_scene_change_value(self, _mafd, _diff) -> np.ndarray:
        if self.is_cython is True:
            res = scd_bp_.calculate_scene_change_value(_mafd=_mafd[1:], _diff=_diff, _min=0.0, _max=1.0)
        else:
            res = np.clip(np.minimum(_mafd[1:], _diff) / 100.0, a_min=0.0, a_max=1.0)

        return res

    def _get_idx_sc(self, _scene_change_val, _threshold, _offset) -> np.ndarray:
        if self.is_cython is True:
            res = scd_bp_.get_idx_sc(_scene_change_val=_scene_change_val, _threshold=_threshold, _offset=_offset)
        else:
            res = np.where(_threshold <= _scene_change_val)[0] + _offset + 2

        return res
