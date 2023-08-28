"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_imgcv.py
Description: A module for human face
"""


import os
import math
import bz2
import cv2
import dlib
import numpy as np
from typing import Optional
from vujade import vujade_download as download_
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_path as path_
from vujade.vujade_debug import printd, pprintd


class DLIB(object):
    def __init__(self, _num_landmarks: int = 68, _spath_model_shape: str = os.path.join(os.getcwd(), 'shape_predictor_68_face_landmarks.dat')) -> None:
        super(DLIB, self).__init__()
        self.num_landmarks = _num_landmarks
        self.url_model_shape = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        self.path_model_shape = path_.Path(_spath_model_shape)

        if self.path_model_shape.path.is_file() is False:
            self._get_model_shape()

        self.detector_hog = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(self.path_model_shape.str)

    def get_detected_faces(self, _ndarr_img: np.ndarray, _img_height_min: int = 256, _img_height_max: int = -1, _dsize_interval: int = 16):
        ndarr_img_height, ndarr_img_width, ndarr_img_channel = _ndarr_img.shape

        if _img_height_min < 1:
            raise ValueError('The _img_height_min should be greater than 0.')

        if _img_height_max < 1:
            _img_height_max = ndarr_img_height

        if _img_height_max < _img_height_min:
            _img_height_max = _img_height_min

        is_detected_faces = False
        res = dlib.rectangles()
        for _idx, _img_height in enumerate(range(_img_height_min, _img_height_max + 1, _dsize_interval)):
            scaling_ratio = _img_height / ndarr_img_height
            ndarr_img_resized = cv2.resize(src=_ndarr_img, dsize=(int(scaling_ratio * ndarr_img_width), int(scaling_ratio * ndarr_img_height)), interpolation=cv2.INTER_LINEAR)
            dets_resized = self.detector_hog(ndarr_img_resized, 1)
            if len(dets_resized) < 1:
                continue
            if len(dets_resized) == 1:
                pt_left, pt_top, pt_right, pt_bottom = self.rect2tuple(_det=dets_resized[0])
                pt_left = int(pt_left / scaling_ratio)
                pt_top = int(pt_top / scaling_ratio)
                pt_right = int(pt_right / scaling_ratio)
                pt_bottom = int(pt_bottom / scaling_ratio)
                res.append(dlib.rectangle(pt_left, pt_top, pt_right, pt_bottom))
                is_detected_faces = True
                break
            else:
                NotImplementedError('The number of detected faces should be 1.')

        if is_detected_faces is False:
            raise ValueError('The number of detected faces should be 1.')

        return res

    def get_landmarks(self, _ndarr_img: np.ndarray, _dets) -> np.ndarray:
        if len(_dets) < 1:
            raise NotImplementedError('The number of detected faces should be greater than 0')

        res = np.zeros((len(_dets), self.num_landmarks, 2), dtype=np.int64)
        for _idx, _det in enumerate(_dets):
            res[_idx, :, :] = np.asarray([(item.x, item.y) for item in self.shape_predictor(_ndarr_img, _det).parts()])

        return res

    @classmethod
    def draw_bboxes(cls, _ndarr_img: np.ndarray, _dets, _color: tuple = (0, 0, 255)) -> None:
        if _ndarr_img.ndim != 3:
            raise ValueError('The _ndarr_img.ndim should be 3.')

        ndarr_img_height, ndarr_img_width, ndarr_img_channel = _ndarr_img.shape

        for _idx, _det in enumerate(_dets):
            if isinstance(_det, (list, tuple, np.ndarray)):
                pt_left, pt_top, pt_right, pt_bottom = _det
            else:
                pt_left, pt_top, pt_right, pt_bottom = cls.rect2tuple(_det=_det)

            pt_left = int(min(max(pt_left, 0), ndarr_img_width - 1))
            pt_top = int(min(max(pt_top, 0), ndarr_img_height - 1))
            pt_right = int(min(max(pt_right, 0), ndarr_img_width - 1))
            pt_bottom = int(min(max(pt_bottom, 0), ndarr_img_height - 1))

            cv2.rectangle(_ndarr_img, pt1=(pt_left, pt_top), pt2=(pt_right, pt_bottom), color=_color)

    @classmethod
    def draw_lmks(cls, _ndarr_img: np.ndarray, _ndarr_lmks: np.ndarray, _radius: int = 1, _color: tuple = (0, 0, 255), _thikness: int = 1) -> None:
        if _ndarr_img.ndim != 3:
            raise ValueError('The _ndarr_img.ndim should be 3.')

        if _ndarr_lmks.ndim != 2:
            raise NotImplementedError('The _ndarr_lmks.ndim should be 2.')

        ndarr_img_height, ndarr_img_width, ndarr_img_channel = _ndarr_img.shape

        for _idx, _lmk in enumerate(_ndarr_lmks):
            lmk_x, lmk_y = _lmk
            lmk_x = int(min(max(lmk_x, 0), ndarr_img_width - 1))
            lmk_y = int(min(max(lmk_y, 0), ndarr_img_height - 1))
            cv2.circle(_ndarr_img, (lmk_x, lmk_y), _radius, _color, _thikness)

    @staticmethod
    def load_image(_spath_image: str, _is_bgr2rgb: bool = True) -> np.ndarray:
        res = cv2.imread(_spath_image, cv2.IMREAD_COLOR)
        if _is_bgr2rgb is True:
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return res

    @ staticmethod
    def lmks68to5(_ndarr_lmks_68: np.ndarray) -> np.ndarray:
        if ((_ndarr_lmks_68.ndim == 2) and (_ndarr_lmks_68.shape[0] == 68)) is False:
            raise NotImplementedError('The _ndarr_lmks_68.shape should be ({68}, 2).')

        res = np.zeros((5, 2), dtype=np.int64)
        res[0, :] = _ndarr_lmks_68[45]
        res[1, :] = _ndarr_lmks_68[42]
        res[2, :] = _ndarr_lmks_68[36]
        res[3, :] = _ndarr_lmks_68[39]
        res[4, :] = _ndarr_lmks_68[33]

        return res

    @staticmethod
    def rect2tuple(_det) -> tuple:
        pt_left = _det.left()
        pt_top = _det.top()
        pt_right = _det.right()
        pt_bottom = _det.bottom()

        return pt_left, pt_top, pt_right, pt_bottom

    @staticmethod
    def tuple2bbox(_pts: tuple) -> np.ndarray:
        if len(_pts) != 4:
            raise ValueError('The len(_pts) should be 4.')

        pt_left, pt_top, pt_right, pt_bottom = _pts
        return np.asarray([[pt_left, pt_top], [pt_right, pt_bottom]], dtype=np.int64)

    @staticmethod
    def bbox2tuple(_ndarr_bbox: np.ndarray) -> tuple:
        if _ndarr_bbox.shape != (2, 2):
            raise NotImplementedError('The _ndarr_lmks.shape should be (2, 2).')

        pt_left = _ndarr_bbox[0][0]
        pt_top = _ndarr_bbox[0][1]
        pt_right = _ndarr_bbox[1][0]
        pt_bottom = _ndarr_bbox[1][1]

        return pt_left, pt_top, pt_right, pt_bottom

    @staticmethod
    def quad2bbox(_ndarr_bbox_quad: np.ndarray) -> np.ndarray:
        if _ndarr_bbox_quad.shape != (4, 2):
            raise ValueError('The _ndarr_bbox_quad.shape should be (4, 2).')
        return _ndarr_bbox_quad[0::2, :]

    @classmethod
    def bbox2quad(cls, _ndarr_bbox: np.ndarray) -> np.ndarray:
        if _ndarr_bbox.shape != (2, 2):
            raise NotImplementedError('The _ndarr_lmks.shape should be (2, 2).')
        pt_left, pt_top, pt_right, pt_bottom = cls.bbox2tuple(_ndarr_bbox=_ndarr_bbox)
        return np.asarray([[pt_left, pt_top], [pt_left, pt_bottom], [pt_right, pt_bottom], [pt_right, pt_top]], dtype=np.int64)

    @classmethod
    def get_bbox_length(cls, _ndarr_bbox: np.ndarray) -> tuple:
        if _ndarr_bbox.shape != (2, 2):
            raise NotImplementedError('The _ndarr_lmks.shape should be (2, 2).')
        pt_left, pt_top, pt_right, pt_bottom = cls.bbox2tuple(_ndarr_bbox=_ndarr_bbox)
        return pt_right - pt_left, pt_bottom - pt_top, math.sqrt(((pt_right - pt_left) ** 2) + ((pt_bottom - pt_top) ** 2))

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

            fp = face_.FaceParsing(_num_landmarks=num_landmarks, _spath_model_shape=spath_model_shape)
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


class FaceAlginmentFFHQ(object):
    """
    Align function from FFHQ dataset pre-processing step
    Reference: https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    """
    @staticmethod
    def run(
            _ndarr_img: np.ndarray,
            _det,
            _ndarr_lmks: np.ndarray,
            _output_size: int = 1024,
            _enable_padding: bool = True,
            _rotate_level: bool = True
            ) -> tuple:
        if _ndarr_lmks.ndim != 2:
            raise NotImplementedError('The _ndarr_lmks.ndim should be 2.')

        if isinstance(_det, (list, tuple)):
            pt_left, pt_top, pt_right, pt_bottom = _det
        else:
            pt_left, pt_top, pt_right, pt_bottom = DLIB.rect2tuple(_det=_det)

        ndarr_bbox = DLIB.tuple2bbox(_pts=(pt_left, pt_top, pt_right, pt_bottom))
        ndarr_lmks = _ndarr_lmks.copy()

        vec_hori_eye_left = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[36, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[39, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        vec_hori_eye_right = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[42, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[45, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        vec_hori_nose = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[31, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[35, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        vec_hori_lip = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[48, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[54, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        rad_eye_left = np.arctan2(vec_hori_eye_left[1], vec_hori_eye_left[0])
        rad_eye_right = np.arctan2(vec_hori_eye_right[1], vec_hori_eye_right[0])
        rad_nose = np.arctan2(vec_hori_nose[1], vec_hori_nose[0])
        rad_lip = np.arctan2(vec_hori_lip[1], vec_hori_lip[0])
        rad = float(np.median((rad_eye_left, rad_eye_right, rad_nose, rad_lip)))

        ndarr_mat_rot = imgcv_.Transform.get_rotation_matrix_2d(_radian=rad, _is_negative=True)
        ndarr_bbox = imgcv_.Transform.rotate_pts_from_anchor(
            _ndarr_src=ndarr_bbox.T,
            _ndarr_anchor=np.mean(ndarr_bbox, axis=0).reshape(-1, 1),
            _ndarr_matrix_rotation=ndarr_mat_rot
        ).astype(np.int64).T

        lm_chin          = ndarr_lmks[0:17, :]  # left-right
        lm_eyebrow_left  = ndarr_lmks[17:22, :] # left-right
        lm_eyebrow_right = ndarr_lmks[22:27, :] # left-right
        lm_nose          = ndarr_lmks[27:31, :] # top-down
        lm_nostrils      = ndarr_lmks[31:36, :] # top-down
        lm_eye_left      = ndarr_lmks[36:42, :] # left-clockwise
        lm_eye_right     = ndarr_lmks[42:48, :] # left-clockwise
        lm_mouth_outer   = ndarr_lmks[48:60, :] # left-clockwise
        lm_mouth_inner   = ndarr_lmks[60:68, :] # left-clockwise

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
        if _rotate_level:
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        else:
            x = np.array([1, 0], dtype=np.float64)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1

        quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y]).astype(np.float32)
        qsize = np.hypot(*x) * 2

        # Shrink.
        shrink = int(np.floor(qsize / _output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(_ndarr_img.shape[1]) / shrink)), int(np.rint(float(_ndarr_img.shape[0]) / shrink)))
            _ndarr_img = cv2.resize(src=_ndarr_img, dsize=rsize, interpolation=cv2.INTER_LINEAR)
            quad /= shrink
            qsize /= shrink
            ndarr_bbox /= shrink
            ndarr_lmks /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, _ndarr_img.shape[1]), min(crop[3] + border, _ndarr_img.shape[0]))
        if crop[2] - crop[0] < _ndarr_img.shape[1] or crop[3] - crop[1] < _ndarr_img.shape[0]:
            _ndarr_img = _ndarr_img[crop[1]:crop[3], crop[0]:crop[2], :]
            quad -= crop[0:2]
            ndarr_bbox -= crop[0:2]
            ndarr_lmks -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - _ndarr_img.shape[1] + border, 0), max(pad[3] - _ndarr_img.shape[0] + border, 0))
        if _enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            _ndarr_img = np.pad(array=_ndarr_img.astype(np.float32), pad_width=((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), mode='reflect')
            h, w, _ = _ndarr_img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            _ndarr_img += (cv2.GaussianBlur(_ndarr_img, ksize=(0, 0), sigmaX=blur, sigmaY=blur) - _ndarr_img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            _ndarr_img += (np.median(_ndarr_img, axis=(0, 1)) - _ndarr_img) * np.clip(mask, 0.0, 1.0)
            _ndarr_img = np.clip(np.rint(_ndarr_img), 0, 255).astype(np.uint8)
            quad += pad[:2]
            ndarr_bbox += pad[:2]
            ndarr_lmks += pad[:2]

        # Transform.
        ndarr_img_aligned, matrix_perspective = imgcv_.Transform.quad(
            _ndarr_img=_ndarr_img,
            _dsize=(_output_size, _output_size),
            _quad_src=(quad + 0.5).astype(np.float32),
            _borderMode=cv2.BORDER_REFLECT
        )
        ndarr_bbox = imgcv_.Transform.warp(_ndarr_pts_src=ndarr_bbox, _matrix=matrix_perspective, _is_normalize=True)
        ndarr_lmks = imgcv_.Transform.warp(_ndarr_pts_src=ndarr_lmks, _matrix=matrix_perspective, _is_normalize=True)

        return ndarr_img_aligned, ndarr_bbox[:2, :].T, ndarr_lmks[:2, :].T


class FaceAlginmenFast(object):
    @classmethod
    def run(
            cls,
            _ndarr_img: np.ndarray,
            _det,
            _ndarr_lmks: np.ndarray,
            _dsize_output: tuple = (1024, 1024),
            _scaling_ratio_quad_offset: float = 0.3,
            _borderValue: tuple = (255, 255, 255)
            ) -> tuple:
        if _ndarr_lmks.ndim != 2:
            raise NotImplementedError('The _ndarr_lmks.ndim should be 2.')

        if isinstance(_det, (list, tuple)):
            pt_left, pt_top, pt_right, pt_bottom = _det
        else:
            pt_left, pt_top, pt_right, pt_bottom = DLIB.rect2tuple(_det=_det)

        ndarr_img_hegiht, ndarr_img_width, ndarr_img_channel = _ndarr_img.shape
        ndarr_bbox = DLIB.tuple2bbox(_pts=(pt_left, pt_top, pt_right, pt_bottom)).astype(np.float32)
        ndarr_bbox_quad = cls._get_scaled_bbox_quad(_ndarr_bbox_quad=DLIB.bbox2quad(_ndarr_bbox=ndarr_bbox), _scaling_ratio_offset=_scaling_ratio_quad_offset)
        ndarr_lmks = _ndarr_lmks.astype(np.float32).copy()

        vec_hori_eye_left = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[36, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[39, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        vec_hori_eye_right = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[42, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[45, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        vec_hori_nose = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[31, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[35, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        vec_hori_lip = imgcv_.Transform.get_ndarr_vector(_ndarr_src=ndarr_lmks[48, :].reshape(1, -1), _ndarr_dst=ndarr_lmks[54, :].reshape(1, -1), _is_left_handed=True).reshape(-1)
        rad_eye_left = np.arctan2(vec_hori_eye_left[1], vec_hori_eye_left[0])
        rad_eye_right = np.arctan2(vec_hori_eye_right[1], vec_hori_eye_right[0])
        rad_nose = np.arctan2(vec_hori_nose[1], vec_hori_nose[0])
        rad_lip = np.arctan2(vec_hori_lip[1], vec_hori_lip[0])
        rad = float(np.median((rad_eye_left, rad_eye_right, rad_nose, rad_lip)))

        ndarr_mat_rot = imgcv_.Transform.get_rotation_matrix_2d(_radian=rad, _is_negative=True)
        ndarr_bbox = imgcv_.Transform.rotate_pts_from_anchor(
            _ndarr_src=ndarr_bbox.T,
            _ndarr_anchor=np.mean(ndarr_bbox, axis=0).reshape(-1, 1),
            _ndarr_matrix_rotation=ndarr_mat_rot
        ).T
        ndarr_bbox_quad = cls._adjust_bbox_quad(
            _ndarr_bbox_quad=imgcv_.Transform.rotate_pts_from_anchor(
                _ndarr_src=ndarr_bbox_quad.T,
                _ndarr_anchor=np.mean(ndarr_bbox_quad, axis=0).reshape(-1, 1),
                _ndarr_matrix_rotation=ndarr_mat_rot
                ).T,
            _ndarr_bbox_quad_center_dst=np.mean((ndarr_lmks[27, :], ndarr_lmks[28, :]), axis=0)
        )

        bbox_length_width, bbox_length_height, bbox_length_diag = DLIB.get_bbox_length(_ndarr_bbox=DLIB.quad2bbox(_ndarr_bbox_quad=ndarr_bbox_quad))
        crop_margin = max(int(bbox_length_width * 0.1), 3)
        ndarr_crop_bbox = (
            max(int(np.floor(np.min(ndarr_bbox_quad[:, 0], axis=0))) - crop_margin, 0),
            max(int(np.floor(np.min(ndarr_bbox_quad[:, 1], axis=0))) - crop_margin, 0),
            min(int(np.ceil(np.max(ndarr_bbox_quad[:, 0], axis=0))) + crop_margin, ndarr_img_width),
            min(int(np.ceil(np.max(ndarr_bbox_quad[:, 1], axis=0))) + crop_margin, ndarr_img_hegiht)
        )
        ndarr_crop_height = ndarr_crop_bbox[3] - ndarr_crop_bbox[1]
        ndarr_crop_width = ndarr_crop_bbox[2] - ndarr_crop_bbox[0]

        if (ndarr_crop_height < ndarr_img_hegiht) or (ndarr_crop_width < ndarr_img_width):
            _ndarr_img = _ndarr_img[ndarr_crop_bbox[1]:ndarr_crop_bbox[3], ndarr_crop_bbox[0]:ndarr_crop_bbox[2], :]
            ndarr_bbox_quad -= ndarr_crop_bbox[0:2]
            ndarr_bbox -= ndarr_crop_bbox[0:2]
            ndarr_lmks -= ndarr_crop_bbox[0:2]

        ndarr_img_aligned, matrix_perspective = imgcv_.Transform.quad(
            _ndarr_img=_ndarr_img,
            _dsize=_dsize_output,
            _quad_src=(ndarr_bbox_quad + 0.5).astype(np.float32),
            _borderMode=cv2.BORDER_CONSTANT,
            _borderValue=_borderValue
        )

        ndarr_bbox = imgcv_.Transform.warp(_ndarr_pts_src=ndarr_bbox, _matrix=matrix_perspective, _is_normalize=True)
        ndarr_lmks = imgcv_.Transform.warp(_ndarr_pts_src=ndarr_lmks, _matrix=matrix_perspective, _is_normalize=True)

        ndarr_bbox = ndarr_bbox[:2, :].T.astype(np.int64)
        ndarr_lmks = ndarr_lmks[:2, :].T.astype(np.int64)

        # Clip
        ndarr_bbox[:, 0] = np.clip(ndarr_bbox[:, 0], a_min=0, a_max=_dsize_output[0] - 1)
        ndarr_bbox[:, 1] = np.clip(ndarr_bbox[:, 1], a_min=0, a_max=_dsize_output[1] - 1)
        ndarr_lmks[:, 0] = np.clip(ndarr_lmks[:, 0], a_min=0, a_max=_dsize_output[0] - 1)
        ndarr_lmks[:, 1] = np.clip(ndarr_lmks[:, 1], a_min=0, a_max=_dsize_output[1] - 1)

        return ndarr_img_aligned, ndarr_bbox, ndarr_lmks

    @staticmethod
    def _get_scaled_bbox_quad(_ndarr_bbox_quad: np.ndarray, _scaling_ratio_offset: float = 1.0) -> np.ndarray:
        _, _, bbox_length_diag = DLIB.get_bbox_length(_ndarr_bbox=DLIB.quad2bbox(_ndarr_bbox_quad=_ndarr_bbox_quad))
        quad_offset = _scaling_ratio_offset * bbox_length_diag
        ndarr_bbox_quad_offset = quad_offset * np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]).astype(np.float32)
        return (_ndarr_bbox_quad + ndarr_bbox_quad_offset).astype(np.float32)

    @staticmethod
    def _adjust_bbox_quad(_ndarr_bbox_quad: np.ndarray, _ndarr_bbox_quad_center_dst: np.ndarray) -> np.ndarray:
        ndarr_bbox_quad_center_src = np.mean(_ndarr_bbox_quad, axis=0)
        ndarr_bbox_quad_offset = _ndarr_bbox_quad_center_dst - ndarr_bbox_quad_center_src
        return _ndarr_bbox_quad + ndarr_bbox_quad_offset
