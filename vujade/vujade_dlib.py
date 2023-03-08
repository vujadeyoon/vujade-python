"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_imgcv.py
Description: A module for dlib
"""


import os
import bz2
import cv2
import dlib
import numpy as np
import scipy.ndimage
import PIL.Image
from typing import Optional
from vujade import vujade_download as download_
from vujade import vujade_path as path_
from vujade.vujade_debug import printd


class DLIB(object):
    def __init__(self, _num_landmarks: int = 68, _spath_model_shape: str = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')) -> None:
        """
        :param _spath_model_shape: path to shape_predictor_68_face_landmarks.dat file
        """
        super(DLIB, self).__init__()
        self.num_landmarks = _num_landmarks
        self.url_model_shape = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        self.path_model_shape = path_.Path(_spath_model_shape)

        if self.path_model_shape.path.is_file() is False:
            self._get_model_shape()

        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(self.path_model_shape.str)

    def load_image(self, _spath_image: str, _is_bgr2rgb: bool = True) -> np.ndarray:
        res = cv2.imread(_spath_image, cv2.IMREAD_COLOR)
        if _is_bgr2rgb is True:
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return res

    def get_detected_faces(self, _ndarr_img: np.ndarray):
        return self.detector(_ndarr_img, 1)

    def get_landmarks(self, _dets, _ndarr_img: np.ndarray) -> np.ndarray:
        if len(_dets) < 1:
            raise ValueError('The number of detected faces should be greater than 0')

        res = np.zeros((len(_dets), self.num_landmarks, 2), dtype=np.int64)
        for _idx, _det in enumerate(_dets):
            res[_idx, :, :] = np.asarray([(item.x, item.y) for item in self.shape_predictor(_ndarr_img, _det).parts()])

        return res

    def _get_model_shape(self) -> None:
        path_model_shape_bz2 = path_.Path(self.path_model_shape.str + '.bz2')

        if path_model_shape_bz2.path.is_file() is False:
            self.path_model_shape.parent.path.mkdir(mode=0o775, parents=True, exist_ok=True)
            download_.Download.run(_url=self.url_model_shape, _spath_filename=path_model_shape_bz2.str)

        data = bz2.BZ2File(path_model_shape_bz2.str).read()
        with open(self.path_model_shape.str, 'wb') as f:
            f.write(data)

        path_model_shape_bz2.unlink(_missing_ok=True)


class FaceParsing(DLIB):
    def __init__(self, _num_landmarks: int = 68, _spath_model_shape: str = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')) -> None:
        super(FaceParsing, self).__init__(_num_landmarks=_num_landmarks, _spath_model_shape=_spath_model_shape)
        self.facial_landmarks_68_idxs = {
            'mouth': (48, 68),
            'inner_mouth': (60, 68),
            'right_eyebrow': (17, 22),
            'left_eyebrow': (22, 27),
            'right_eye': (36, 42),
            'left_eye': (42, 48),
            'nose': (27, 36),
            'jaw': (0, 17),
        }
        self.facial_landmarks_idxs = self.facial_landmarks_68_idxs
        self.colors = [
            (19, 199, 109),
            (79, 76, 240),
            (230, 159, 23),
            (168, 100, 168),
            (158, 163, 32),
            (163, 38, 32),
            (180, 42, 220),
            (0, 0, 255)
        ]

    def run(self, _ndarr_lmks: np.ndarray, _img_height: int, _img_width: int) -> dict:
        """
        Reference: https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/helpers.py

        Usage:
            num_landmarks = 68
            spath_model_shape = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')
            spath_image = os.path.join(os.getcwd(), 'img_src.png')

            fp = dlib_.FaceParsing(_num_landmarks=num_landmarks, _spath_model_shape=spath_model_shape)
            ndarr_img_bgr = fp.load_image(_spath_image=spath_image, _is_bgr2rgb=False)
            dets = fp.get_detected_faces(_ndarr_img=ndarr_img_bgr)
            if len(dets) == 1:
                ndarr_lmks = fp.get_landmarks(_dets=dets, _ndarr_img=ndarr_img_bgr)[0]
                lmks_face_parsed = fp.run(_ndarr_lmks=ndarr_lmks, _img_height=ndarr_img_bgr.shape[0], _img_width=ndarr_img_bgr.shape[1])
        """

        res = {_key: None for _key in self.facial_landmarks_68_idxs.keys()}

        for (i, name) in enumerate(self.facial_landmarks_idxs.keys()):
            # grab the (x, y)-coordinates associated with the
            # face landmark
            (j, k) = self.facial_landmarks_idxs[name]
            pts = _ndarr_lmks[j:k]

            # check if are supposed to draw the jawline
            if name == "jaw":
                res[name] = pts
            else:
                hull = cv2.convexHull(pts)
                overlay = np.zeros(shape=(_img_height, _img_width), dtype=np.uint8)
                cv2.drawContours(overlay, [hull], -1, 255, -1)
                pts_y, pts_x = np.where(overlay == 255)
                res[name] = np.asarray([(_pts_y, _pts_x) for _idx, (_pts_y, _pts_x) in enumerate(zip(pts_y, pts_x))])

        return res

    def render(self, _ndarr_img_src: np.ndarray, _lmks_face_parsed: dict, _colors: Optional[list] = None, _alpha: float = 0.75) -> np.ndarray:
        """
        Usage:
            ndarr_img_face_parsed = fp.render(_ndarr_img_src=ndarr_img_bgr, _lmks_face_parsed=lmks_face_parsed)
            cv2.imwrite('img_face_parsed.png', ndarr_img_face_parsed)
        """
        if _colors is None:
            _colors = self.colors

        if len(_lmks_face_parsed) != len(_colors):
            raise ValueError('Both lengths of the _lmks_face_parsed and _colors should be same.')

        ndarr_img_overay = _ndarr_img_src.copy()
        ndarr_img_render = _ndarr_img_src.copy()

        for _idx, (_name, _ndarr_lmks_face_parsed) in enumerate(_lmks_face_parsed.items()):
            if _name == "jaw":
                # Since the jawline is a non-enclosed facial region, just draw lines between the (x, y)-coordinates.
                for _l in range(1, len(_ndarr_lmks_face_parsed)):
                    pts_1 = tuple(_ndarr_lmks_face_parsed[_l - 1])
                    pts_2 = tuple(_ndarr_lmks_face_parsed[_l])
                    cv2.line(ndarr_img_overay, pts_1, pts_2, _colors[_idx], 2)
            else:
                for _idy, (_pts_y, _pts_x) in enumerate(_ndarr_lmks_face_parsed):
                    ndarr_img_overay[_pts_y, _pts_x, :] = _colors[_idx]

        # Apply the transparent overlay.
        cv2.addWeighted(ndarr_img_overay, _alpha, ndarr_img_render, 1.0 - _alpha, 0, ndarr_img_render)

        return ndarr_img_render


class FaceAlginment(DLIB):
    def __init__(self, _num_landmarks: int = 68, _spath_model_shape: str = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')) -> None:
        super(FaceAlginment, self).__init__(_num_landmarks=_num_landmarks, _spath_model_shape=_spath_model_shape)

    def run(self,
            _spath_img: str,
            _ndarr_lmks: np.ndarray,
            _output_size: int = 1024,
            _transform_size: int = 4096,
            _enable_padding: bool = True,
            _x_scale: int = 1,
            _y_scale: int = 1,
            _em_scale: float = 0.1,
            _alpha: bool = False,
            _resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC
            ) -> np.ndarray:
        """
        Align function from FFHQ dataset pre-processing step
        https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        Usage:
            num_landmarks = 68
            spath_model_shape = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')
            spath_image = os.path.join(os.getcwd(), 'img_src.png')

            fa = FaceAlginment(_num_landmarks=num_landmarks, _spath_model_shape=spath_model_shape)
            ndarr_img_bgr = fa.load_image(_spath_image=spath_image, _is_bgr2rgb=False)
            dets = fa.get_detected_faces(_ndarr_img=ndarr_img_bgr)
            if len(dets) == 1:
                ndarr_lmks = fa.get_landmarks(_dets=dets, _ndarr_img=ndarr_img_bgr)
                ndarr_img_aligned = fa.run(_spath_img=spath_image, _ndarr_lmks=ndarr_lmks)
                cv2.imwrite('./img_aligned.png', ndarr_img_aligned)
        """

        if _ndarr_lmks.shape[0] != 1:
            raise NotImplementedError('The face alignment for multiple faces has not been supported yet.')

        lm = _ndarr_lmks[0]
        lm_chin          = lm[0 :17]  # left-right
        lm_eyebrow_left  = lm[17:22]  # left-right
        lm_eyebrow_right = lm[22:27]  # left-right
        lm_nose          = lm[27:31]  # top-down
        lm_nostrils      = lm[31:36]  # top-down
        lm_eye_left      = lm[36:42]  # left-clockwise
        lm_eye_right     = lm[42:48]  # left-clockwise
        lm_mouth_outer   = lm[48:60]  # left-clockwise
        lm_mouth_inner   = lm[60:68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= _x_scale
        y = np.flipud(x) * [-_y_scale, _y_scale]
        c = eye_avg + eye_to_mouth * _em_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # Load in-the-wild image.
        if not os.path.isfile(_spath_img):
            raise FileNotFoundError("Cannot find source image. Please run '--wilds' before '--align'.")
        img = PIL.Image.open(_spath_img).convert('RGBA').convert('RGB')

        # Shrink.
        shrink = int(np.floor(qsize / _output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if _enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.uint8(np.clip(np.rint(img), 0, 255))
            if _alpha:
                mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
                mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
                img = np.concatenate((img, mask), axis=2)
                img = PIL.Image.fromarray(img, 'RGBA')
            else:
                img = PIL.Image.fromarray(img, 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((_transform_size, _transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if _output_size < _transform_size:
            img = img.resize(size=(_output_size, _output_size), resample=_resample)

        ndarr_img_aligned = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        return ndarr_img_aligned
