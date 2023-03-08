"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_segmentation.py
Description: A module for segmentation
"""


import cv2
import numpy as np
from typing import Optional, List
from vujade import vujade_imgcv as imgcv_
from vujade.vujade_debug import printd


class Visualize(object):
    def __init__(
            self,
            _color_code: Optional[list] = None,
            _alpha: Optional[float] = 0.6,
            _class_background: int = 0,
            _class_ignore: int = 255,
            _is_rgb2bgr: bool = False,
            _dtype: type = np.uint8,
            _bbox_color: tuple = (0, 0, 255),
            _bbox_thickness: int = 2
    ) -> None:
        super(Visualize, self).__init__()
        self.color_code = _color_code
        self.alpha = _alpha
        self.class_background = _class_background
        self.class_ignore = _class_ignore
        self.is_rgb2bgr = _is_rgb2bgr
        self.dtype = _dtype
        self.bbox_color = _bbox_color
        self.bbox_thickness = _bbox_thickness
        self._check_valid()

    def overlay(self, _ndarr_img: np.ndarray, _ndarr_mask: np.ndarray, _bbox: Optional[list] = None) -> np.ndarray:
        if (_ndarr_img.ndim != 3) or (_ndarr_mask.ndim != 2):
            raise ValueError('The dimension of the ndarray may be not correct.')

        if (_ndarr_img.shape[0] != _ndarr_mask.shape[0]) or (_ndarr_img.shape[1] != _ndarr_mask.shape[1]):
            raise ValueError('The both resolution of the ndarrays should be same.')

        ndarr_mask_color = self.mask2color(_ndarr_mask=_ndarr_mask)
        ndarr_overlay = cv2.addWeighted(_ndarr_img, 1 - self.alpha, ndarr_mask_color, self.alpha, 0)

        if imgcv_.check_valid_pixel(_x=self.class_ignore, _bit=8) is True:
            idx_target = (_ndarr_mask == self.class_ignore)
            ndarr_overlay[idx_target] = _ndarr_img[idx_target]

        if _bbox is not None:
            self._bbox(_ndarr_img=ndarr_overlay, _bbox=_bbox, _color=self.bbox_color, _thickness=self.bbox_thickness)

        return ndarr_overlay

    def mask2color(self, _ndarr_mask: np.ndarray) -> np.ndarray:
        res = np.zeros((_ndarr_mask.shape[0], _ndarr_mask.shape[1], 3), dtype=self.dtype)

        for _idx, _color_code in enumerate(self.color_code):
            class_name = _color_code['name'].lower()
            class_color = _color_code['bgr']

            # Reverse color order because the OpenCV supports bgr order, not rgb order.
            if self.is_rgb2bgr is True:
                class_color = class_color[::-1]

            # Calculate idx_class
            if class_name in {'bg', 'background'}:
                idx_class = self.class_background
            elif class_name in {'ignore'}:
                idx_class = self.class_ignore
            else:
                idx_class = _idx

            res[(_ndarr_mask == idx_class)] = class_color

        return res

    def _check_valid(self):
        for _idx, _color_code in enumerate(self.color_code):
            class_name = _color_code['name'].lower()
            if class_name in {'bg', 'background'}:
                if imgcv_.check_valid_pixel(_x=self.class_background, _bit=8) is False:
                    raise ValueError('The self.class_background, {} may be incorrect.'.format(self.class_background))
            elif class_name in {'ignore'}:
                if imgcv_.check_valid_pixel(_x=self.class_ignore, _bit=8) is False:
                    raise ValueError('The self.class_ignore, {} may be incorrect.'.format(self.class_ignore))
            else:
                pass

    def _bbox(self, _ndarr_img: np.ndarray, _bbox: list, _color: tuple, _thickness: int = 2) -> None:
        cv2.rectangle(_ndarr_img, (int(_bbox[0]), int(_bbox[1])), (int(_bbox[2]), int(_bbox[3])), color=_color, thickness=_thickness)


def get_color_maps() -> dict:
    cmap = dict()  # Color order: BGR
    cmap['CelebAMask-HQ'] = {
        'background': (0, 0, 0),
        'skin': (0, 0, 204),
        'l_brow': (255, 255, 0),
        'r_brow': (255, 255, 51),
        'l_eye': (255, 51, 51),
        'r_eye': (204, 0, 204),
        'eye_g': (0, 204, 204),
        'l_ear': (0, 51, 102),
        'r_ear': (0, 0, 255),
        'ear_r': (204, 204, 0),
        'nose': (0, 153, 76),
        'mouth': (0, 204, 102),
        'u_lip': (0, 255, 255),
        'l_lip': (153, 0, 0),
        'neck': (51, 153, 255),
        'neck_l': (0, 51, 0),
        'cloth': (0, 204, 0),
        'hair': (204, 0, 0),
        'hat': (153, 51, 255)
    }

    cmap['LaPa'] = {
        'background': (0, 0, 0),
        'skin': (255, 153, 0),
        'left_eyebrow': (153, 255, 102),
        'right_eyebrow': (153, 204, 0),
        'left_eye': (102, 255, 255),
        'right_eye': (204, 255, 255),
        'nose': (0, 153, 255),
        'upper lip': (255, 102, 255),
        'inner mouth': (51, 0, 102),
        'lower lip': (255, 204, 255),
        'hair': (102, 0, 255)
    }

    cmap['FaceSynthetics'] = {
        'BACKGROUND': (180, 120, 30),
        'SKIN': (230, 200, 170),
        'NOSE': (40, 40, 210),
        'RIGHT_EYE': (135, 220, 150),
        'LEFT_EYE': (45, 160, 40),
        'RIGHT_BROW': (120, 190, 250),
        'LEFT_BROW': (20, 128, 250),
        'RIGHT_EAR': (220, 220, 130),
        'LEFT_EAR': (190, 190, 0),
        'MOUTH_INTERIOR': (160, 95, 145),
        'TOP_LIP': (135, 135, 255),
        'BOTTOM_LIP': (195, 165, 195),
        'NECK': (194, 118, 228),
        'HAIR': (65, 75, 115),
        'BEARD': (128, 128, 128),
        'CLOTHING': (200, 200, 200),
        'GLASSES': (128, 85, 25),
        'HEADWEAR': (140, 218, 217),
        'FACEWEAR': (60, 186, 180),
        'IGNORE': (0, 0, 0)
    }

    return cmap


def get_color_code(_name_dataset: str) -> List[dict]:
    cmap = get_color_maps()

    if not _name_dataset in cmap.keys():
        raise NotImplementedError

    res = list()
    for _idx, (_name, _color_bgr) in enumerate(cmap[_name_dataset].items()):
        res.append({'name': _name, 'bgr': _color_bgr})

    return res
