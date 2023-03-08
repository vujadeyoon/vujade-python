"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: face_parsing.py
Description: A transformation module for face_parsing

Acknowledgement:
    1. This implementation is highly inspired from the GitHub repository, https://github.com/zllrunning/face-parsing.PyTorch.
"""


import random
import PIL
import cv2
import torch
import numpy as np
import albumentations as A
import albumentations.augmentations.geometric.functional as AF
import torchvision.transforms.functional as F
from typing import Optional, Union
from external_lib.iBUG.roi_tanh_warping import wrapper
from vujade import vujade_list as list_
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printd


class Compose(object):
    # Composes several co_transforms together.
    # For example:
    # >>> co_transforms.Compose([
    # >>>     co_transforms.CenterCrop(10),
    # >>>     co_transforms.ToTensor(),
    # >>>  ])

    def __init__(self, _co_transforms: list) -> None:
        self.co_transforms = _co_transforms

    def __call__(self, _image: object, _mask: object, _bbox: object) -> tuple:
        for trans in self.co_transforms:
            _image, _mask, _bbox = trans(_image, _mask, _bbox)

        return _image, _mask, _bbox


class OneOf:
    def __init__(self, _co_transforms: list, _prob: float = 0.5) -> None:
        self.co_transforms = _co_transforms
        self.prob = _prob

    def __call__(self, _image: object, _mask: object, _bbox: object) -> tuple:
        if (random.random() < self.prob) and (self.co_transforms): # Not empty
            trans = random.choice(self.co_transforms)
            _image, _mask, _bbox = trans(_image, _mask, _bbox)

        return _image, _mask, _bbox


class RandomCrop(object):
    def __init__(self, _dsize: tuple, _prob: float = 0.5) -> None:
        self.dsize = _dsize
        self.prob = _prob

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            _image, _mask, _bbox = self._run(_image=_image, _mask=_mask, _bbox=_bbox)

        return _image, _mask, _bbox

    def _run(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        assert _image.size == _mask.size
        W, H = self.dsize
        w, h = _image.size

        if (W, H) == (w, h):
            return _image, _mask, _bbox
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            _image = _image.resize((w, h), PIL.Image.BILINEAR)
            _mask = _mask.resize((w, h), PIL.Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H

        return _image.crop(crop), _mask.crop(crop), self._refine_bbox(_bbox=_bbox, _crop=crop)

    def _refine_bbox(self, _bbox: np.ndarray, _crop: tuple) -> np.ndarray:
        if _bbox[0] < _crop[0]:
            _bbox[0] = _crop[0]
        if _bbox[1] < _crop[1]:
            _bbox[1] = _crop[1]
        if _crop[2] < _bbox[2]:
            _bbox[2] = _crop[2]
        if _crop[3] < _bbox[3]:
            _bbox[3] = _crop[3]

        return _bbox


class HorizontalFlip(object):
    def __init__(self, _pair_flip_horizontal: tuple = ([2, 3], [4, 5], [7, 8]), _prob: float = 0.5) -> None:
        self.pair_flip_horizontal = _pair_flip_horizontal
        self.prob = _prob

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            _image, _mask, _bbox = self._run(_image=_image, _mask=_mask, _bbox=_bbox)

        return _image, _mask, _bbox

    def _run(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        ndarr_mask = np.asarray(_mask)
        ndarr_palette = ndarr_mask.copy()
        for _idx, _pair in enumerate(self.pair_flip_horizontal):
            ndarr_palette[ndarr_mask == _pair[0]] = _pair[1]
            ndarr_palette[ndarr_mask == _pair[1]] = _pair[0]
        target = PIL.Image.fromarray(ndarr_palette)

        return _image.transpose(PIL.Image.FLIP_LEFT_RIGHT), target.transpose(PIL.Image.FLIP_LEFT_RIGHT), self._refine_bbox(_bbox=_bbox, _dsize=_image.size)

    def _refine_bbox(self, _bbox: np.ndarray, _dsize: tuple) -> np.ndarray:
        bbox = _bbox.copy()
        width = _dsize[0]
        bbox[0] = (width - 1) - _bbox[2]
        bbox[2] = (width - 1) - _bbox[0]
        return bbox


class VerticalFlip(object):
    def __init__(self, _pair_flip_horizontal: tuple = ([2, 3], [4, 5], [7, 8]), _prob: float = 0.5) -> None:
        self.pair_flip_horizontal = _pair_flip_horizontal
        self.prob = _prob

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            _image, _mask, _bbox = self._run(_image=_image, _mask=_mask, _bbox=_bbox)

        return _image, _mask, _bbox

    def _run(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        ndarr_mask = np.asarray(_mask)
        ndarr_palette = ndarr_mask.copy()
        for _idx, _pair in enumerate(self.pair_flip_horizontal):
            ndarr_palette[ndarr_mask == _pair[0]] = _pair[1]
            ndarr_palette[ndarr_mask == _pair[1]] = _pair[0]
        target = PIL.Image.fromarray(ndarr_palette)

        return _image.transpose(PIL.Image.FLIP_TOP_BOTTOM), target.transpose(PIL.Image.FLIP_TOP_BOTTOM), self._refine_bbox(_bbox=_bbox, _dsize=_image.size)

    def _refine_bbox(self, _bbox: np.ndarray, _dsize: tuple) -> np.ndarray:
        bbox = _bbox.copy()
        height = _dsize[1]
        bbox[1] = (height - 1) - _bbox[3]
        bbox[3] = (height - 1) - _bbox[1]
        return bbox


class Rotate90(object):
    def __init__(self, _pair_flip_horizontal: tuple = ([2, 3], [4, 5], [7, 8]), _prob: float = 0.5) -> None:
        self.pair_flip_horizontal = _pair_flip_horizontal
        self.prob = _prob

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            _image, _mask, _bbox = self._run(_image=_image, _mask=_mask, _bbox=_bbox)

        return _image, _mask, _bbox

    def _run(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        # ndarr_mask = np.asarray(_mask)
        # ndarr_palette = ndarr_mask.copy()
        # for _idx, _pair in enumerate(self.pair_flip_horizontal):
        #     ndarr_palette[ndarr_mask == _pair[0]] = _pair[1]
        #     ndarr_palette[ndarr_mask == _pair[1]] = _pair[0]
        # target = PIL.Image.fromarray(ndarr_palette)

        return _image.rotate(90), _mask.rotate(90), self._refine_bbox(_bbox=_bbox, _dsize=_image.size)

    def _refine_bbox(self, _bbox: np.ndarray, _dsize: tuple) -> np.ndarray:
        bbox = np.zeros_like(_bbox)
        width, height = _dsize[0], _dsize[1]
        bbox[0] = _bbox[1]
        bbox[1] = width - _bbox[2]
        bbox[2] = _bbox[3]
        bbox[3] = width - _bbox[0]
        return bbox


class RandomScale(object):
    def __init__(self, _scale: tuple = (1, ), _prob: float = 0.5) -> None:
        self.scale = _scale
        self.prob = _prob

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            _image, _mask, _bbox = self._run(_image=_image, _mask=_mask, _bbox=_bbox)

        return _image, _mask, _bbox

    def _run(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        W, H = _image.size
        scale = random.choice(self.scale)
        w, h = int(W * scale), int(H * scale)

        return _image.resize((w, h), PIL.Image.BILINEAR), _mask.resize((w, h), PIL.Image.NEAREST), scale * _bbox


class RandomShiftScaleRotate(object):
    def __init__(self, _degree: tuple = (-0, 0), _prob: float = 0.5) -> None:
        self.degree = _degree
        self.prob = _prob
        self.trans = A.Compose(
            [A.ShiftScaleRotate(shift_limit=0.0625,
                                scale_limit=0.1,
                                rotate_limit=45,
                                p=1.0)],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
        )

    def __call__(self, _image: np.ndarray, _mask: np.ndarray, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            transformed = self.trans(image=_image, mask=_mask, bboxes=[_bbox.tolist()], category_ids=[0])
            _image = transformed['image'].astype(_image.dtype)
            _mask = transformed['mask'].astype(_mask.dtype)
            _bbox = np.asarray(transformed['bboxes'][0]).astype(_bbox.dtype)

        return _image, _mask, _bbox


class ColorJitter(object):
    def __init__(self, _brightness: float = 0.5, _contrast: float = 0.5, _saturation: float = 0.5, _prob: float = 0.5):
        if (_brightness <= 0.0) and (_contrast <= 0.0) and (_saturation <= 0.0):
            raise ValueError
        self.brightness = self._get_values(_val=_brightness)
        self.contrast = self._get_values(_val=_contrast)
        self.saturation = self._get_values(_val=_saturation)
        self.prob = _prob

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            _image = self._run(_image=_image)

        return _image, _mask, _bbox

    def _run(self, _image: PIL.Image.Image) -> tuple:
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        _image = PIL.ImageEnhance.Brightness(_image).enhance(r_brightness)
        _image = PIL.ImageEnhance.Contrast(_image).enhance(r_contrast)
        _image = PIL.ImageEnhance.Color(_image).enhance(r_saturation)

        return _image

    def _get_values(self, _val: float) -> tuple:
        return (max(0.0, 1.0 - _val), 1.0 + _val)


class Resize(object):
    def __init__(self, _dsize: tuple, _prob: float = 0.5) -> None:
        self.dsize = _dsize
        self.prob = _prob

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            dsize_src = _image.size
            _image = self._run(_image=_image, _resample=PIL.Image.BILINEAR)
            _mask = self._run(_image=_mask, _resample=PIL.Image.NEAREST)
            _bbox = self._refine_bbox(_bbox=_bbox, _dsize_src=dsize_src)

        return _image, _mask, _bbox

    def _run(self, _image: PIL.Image.Image, _resample: int) -> PIL.Image.Image:
        if _image.size == self.dsize:
            res = _image
        else:
            res = _image.resize(self.dsize, _resample)

        return res

    def _refine_bbox(self, _bbox: np.ndarray, _dsize_src: tuple) -> np.ndarray:
        ndarr_dsize_src = np.asarray(_dsize_src)
        ndarr_dsize_dst = np.asarray(self.dsize)
        ndarr_scale = ndarr_dsize_dst / ndarr_dsize_src

        return (ndarr_scale * _bbox.reshape(-1, 2)).flatten()


class ToNdarrs(object):
    def __init__(self, _code: tuple = (cv2.COLOR_RGB2BGR, None), _dtype: tuple = (np.float32, np.int64)) -> None:
        self.code = _code
        self.dtype = _dtype

    def __call__(self, _image: PIL.Image.Image, _mask: PIL.Image.Image, _bbox: np.ndarray) -> tuple:
        return self.run(_image=_image, _code=self.code[0], _dtype=self.dtype[0]), self.run(_image=_mask, _code=self.code[1], _dtype=self.dtype[1]), _bbox

    def run(self, _image: PIL.Image.Image, _code: Optional[int] = cv2.COLOR_RGB2BGR, _dtype: type = np.float32) -> np.ndarray:
        res = np.asarray(_image, dtype=_dtype)

        if _code is not None:
            res = cv2.cvtColor(src=res, code=_code)

        return res


class RoiWarpNumpy(object):
    def __init__(self, _model: str = 'roi_tanh_polar', _dsize_polar: tuple = (512, 512)) -> None:
        self.model = _model
        self.dsize_polar = _dsize_polar
        self.roi_warp_np = wrapper.RoIWarpNumpy(_model=self.model, _dsize_polar=self.dsize_polar)

    def __call__(self, _image: np.ndarray, _mask: np.ndarray, _bbox: np.ndarray) -> tuple:
        if _mask.ndim != 3:
            target = np.zeros((*_mask.shape, 3), dtype=_image.dtype)
            target[:, :, 0] = _mask
        else:
            target = _mask
        return self.roi_warp_np.forward(_ndarr_cartesian=_image, _bbox_face=_bbox), \
               self.roi_warp_np.forward(_ndarr_cartesian=target, _bbox_face=_bbox)[:, :, 0].astype(_mask.dtype), \
               _bbox


class RoIEnhancement(object):
    def __init__(self, _scaling_meshg: Optional[float] = 2.0, _prob: float = 0.5):
        super(RoIEnhancement, self).__init__()
        self.scaling_meshg = _scaling_meshg
        self.prob = _prob

    def __call__(self, _image: np.ndarray, _mask: np.ndarray, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            _image = self.run(_image=_image, _img_dsize=_image.shape[:2][::-1], _bbox=_bbox)

        return _image, _mask, _bbox

    def run(self, _image: np.ndarray, _img_dsize: tuple, _bbox: np.ndarray):
        meshgrid = self._get_meshgrid_2(_img_dsize=_img_dsize, _bbox=_bbox)
        if (isinstance(meshgrid, np.ndarray) is True) and (self.scaling_meshg is not None):
            meshgrid = self._normalize(_meshgrid=meshgrid)
        weight = self._get_weight(_meshgrid=meshgrid)
        res = _image * weight.reshape(*weight.shape, -1)

        return res

    def _normalize(self, _meshgrid: np.ndarray) -> np.ndarray:
        meshg_max, meshg_min = np.max(_meshgrid), np.min(_meshgrid)
        return self.scaling_meshg * (_meshgrid - meshg_min) / meshg_max

    def _get_weight(self, _meshgrid: Union[np.ndarray, float]) -> np.ndarray:
        mu = 0.0
        std = 1.0
        scaling_factor = (std * np.sqrt(2.0 * np.pi))
        res = scaling_factor * np.exp((((_meshgrid - mu) / std) ** 2) / -2.0) / (std * np.sqrt(2.0 * np.pi))

        return res

    def _get_meshgrid_1(self, _img_dsize: tuple, _bbox: np.ndarray) -> np.ndarray:
        bbox = _bbox.astype(np.int64)
        ndarr_template = np.zeros(_img_dsize[::-1], dtype=np.uint8)
        ndarr_template[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

        distance_map = cv2.distanceTransform(ndarr_template, cv2.DIST_L2,
                                             cv2.DIST_MASK_5)  # Option: cv2.DIST_MASK_PRECISE
        res = np.fabs(distance_map.max() - distance_map)

        return res

    def _get_meshgrid_2(self, _img_dsize: tuple, _bbox: np.ndarray) -> np.ndarray:
        img_width, img_hegiht = _img_dsize
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _bbox.astype(np.int64)
        bbox_width, bbox_height = bbox_x2 - bbox_x1 + 1, bbox_y2 - bbox_y1 + 1

        margin_x_1, margin_x_2 = bbox_x1, (img_width - 1) - bbox_x2
        margin_y_1, margin_y_2 = bbox_y1, (img_hegiht - 1) - bbox_y2
        margin_x, margin_y = max(margin_x_1, margin_x_2), max(margin_y_1, margin_y_2)
        meshg_width, meshg_height = bbox_width + int(2.0 * margin_x), bbox_height + int(2.0 * margin_y)

        meshg_x, meshg_y = np.meshgrid(
            np.linspace(-int(meshg_width / 2.0), int(meshg_width / 2.0), meshg_width + 1),
            np.linspace(-int(meshg_height / 2.0), int(meshg_height / 2.0), meshg_height + 1)
        )
        dist_xy = np.sqrt(meshg_x ** 2 + meshg_y ** 2)

        offset_y = abs(margin_y_2 - margin_y_1)
        offset_x = abs(margin_x_2 - margin_x_1)

        if margin_y_1 < margin_y_2:
            if margin_x_1 < margin_x_2:
                res = dist_xy[offset_y:(offset_y + img_hegiht), offset_x:(offset_x + img_width)]
            else:
                res = dist_xy[offset_y:(offset_y + img_hegiht), :img_width]
        else:
            if margin_x_1 < margin_x_2:
                res = dist_xy[:img_hegiht, offset_x:(offset_x + img_width)]
            else:
                res = dist_xy[:img_hegiht, :img_width]

        return res


class GridDistortion(object):
    """
    http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    def __init__(self, _num_steps: int = 5, _distort_limit: float = 0.3, _prob: float = 0.5) -> None:
        super(GridDistortion, self).__init__()
        self.num_steps = _num_steps
        self.distort_limit = _distort_limit
        self.prob = _prob

    def __call__(self, _image: np.ndarray, _mask: np.ndarray, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            img_height, img_width = _image.shape[:2]
            xx = self._get_grid(_length=img_width)
            yy = self._get_grid(_length=img_height)
            _image, _mask = self._get_grid_distort(_image=_image, _mask=_mask, _xx=xx, _yy=yy)
            _bbox = self._refine_bbox(_bbox=_bbox, _xx=xx, _yy=yy)

        return _image, _mask, _bbox

    def _get_grid(self, _length: int) -> np.ndarray:
        step = _length // self.num_steps
        res = np.zeros(_length, np.float32)
        prev = 0
        for _idx in range(0, _length, step):
            start = _idx
            end = _idx + step
            if end > _length:
                end = _length
                cur = _length
            else:
                cur = prev + step * (1 + random.uniform(-self.distort_limit, self.distort_limit))

            res[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        return res

    def _get_grid_distort(self, _image: np.ndarray, _mask: np.ndarray, _xx: np.ndarray, _yy: np.ndarray) -> tuple:
        map_x, map_y = np.meshgrid(_xx, _yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        ndarr_img = cv2.remap(_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        if _mask is not None:
            ndarr_mask = cv2.remap(_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        else:
            ndarr_mask = _mask

        return ndarr_img, ndarr_mask

    def _refine_bbox(self, _bbox: np.ndarray, _xx: np.ndarray, _yy: np.ndarray) -> np.ndarray:
        _bbox[0] = np.abs(_xx - _bbox[0]).argmin()
        _bbox[1] = np.abs(_yy - _bbox[1]).argmin()
        _bbox[2] = np.abs(_xx - _bbox[2]).argmin()
        _bbox[3] = np.abs(_yy - _bbox[3]).argmin()

        return _bbox


class RoIGridDistortion(object):
    def __init__(self, _margin: tuple = (30, 30), _labels: tuple = (1, ), _prob: float = 0.5) -> None:
        super(RoIGridDistortion, self).__init__()
        self.margin = _margin
        self.labels = _labels
        self.prob = _prob

    def __call__(self, _image: np.ndarray, _mask: np.ndarray, _bbox: np.ndarray) -> tuple:
        if random.random() < self.prob:
            label = random.choice(self.labels)
            img_height, img_width = _image.shape[:2]
            list_x_prev_curr, list_x_start_end, list_y_prev_curr, list_y_start_end = self._get_list(_mask=_mask, _size=_image.shape[:2], _label=label)
            xx = self._get_grid(_length=img_width, _ndarr_prev_curr=list_x_prev_curr, _ndarr_start_end=list_x_start_end)
            yy = self._get_grid(_length=img_height, _ndarr_prev_curr=list_y_prev_curr, _ndarr_start_end=list_y_start_end)
            _image, _mask = self._get_grid_distort(_image=_image, _mask=_mask, _xx=xx, _yy=yy)
            _bbox = self._refine_bbox(_bbox=_bbox, _xx=xx, _yy=yy)

        return _image, _mask, _bbox

    def _get_list(self, _mask: np.ndarray, _size: tuple, _label: int) -> tuple:
        target_x, target_y = np.where(np.transpose(_mask, (1, 0)) == _label)
        is_empty = (target_x.size == 0)
        if is_empty is True:
            target_x_min, target_x_max = 0, 0
            target_y_min, target_y_max = 0, 0
            margin_x, margin_y = 0, 0
        else:
            target_x_min, target_x_max = target_x.min(), target_x.max()
            target_y_min, target_y_max = target_y.min(), target_y.max()
            margin_x, margin_y = self.margin[0], self.margin[1]

        ndarr_x_prev_curr = np.asarray([0, target_x_min, target_x_max, _size[1] - 1]).clip(min=0.0, max=_size[1] - 1).astype(np.int64)
        ndarr_x_start_end = np.asarray([0, target_x_min - margin_x, target_x_max + margin_x, _size[1] - 1]).clip(min=0.0, max=_size[1] - 1).astype(np.int64)
        ndarr_y_prev_curr = np.asarray([0, target_y_min, target_y_max, _size[0] - 1]).clip(min=0.0, max=_size[0] - 1).astype(np.int64)
        ndarr_y_start_end = np.asarray([0, target_y_min - margin_y, target_y_max + margin_y, _size[0] - 1]).clip(min=0.0, max=_size[0] - 1).astype(np.int64)

        return ndarr_x_prev_curr, ndarr_x_start_end, ndarr_y_prev_curr, ndarr_y_start_end

    def _get_grid(self, _length: int, _ndarr_prev_curr: np.ndarray, _ndarr_start_end: np.ndarray) -> np.ndarray:
        res = np.zeros(_length, np.float32)
        for _idx in range(len(_ndarr_prev_curr) - 1):
            start = _ndarr_start_end[_idx]
            end = _ndarr_start_end[_idx + 1]
            prev = _ndarr_prev_curr[_idx]
            curr = _ndarr_prev_curr[_idx + 1]
            res[start:end] = np.linspace(prev, curr, end - start)

        return res

    def _get_grid_distort(self, _image: np.ndarray, _mask: np.ndarray, _xx: np.ndarray, _yy: np.ndarray) -> tuple:
        map_x, map_y = np.meshgrid(_xx, _yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        ndarr_img = cv2.remap(_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        ndarr_mask = cv2.remap(_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        return ndarr_img, ndarr_mask

    def _refine_bbox(self, _bbox: np.ndarray, _xx: np.ndarray, _yy: np.ndarray) -> np.ndarray:
        _bbox[0] = np.abs(_xx - _bbox[0]).argmin()
        _bbox[1] = np.abs(_yy - _bbox[1]).argmin()
        _bbox[2] = np.abs(_xx - _bbox[2]).argmin()
        _bbox[3] = np.abs(_yy - _bbox[3]).argmin()

        return _bbox


class ToTensors(object):
    def __init__(self, _dtype_image: type = torch.float32, _dtype_mask: type = torch.int64, _dtype_bbox: type = torch.float32) -> None:
        self.dtype_image = _dtype_image
        self.dtype_mask = _dtype_mask
        self.dtype_bbox = _dtype_bbox

    def __call__(self, _image: np.ndarray, _mask: np.ndarray, _bbox: np.ndarray) -> tuple:
        return self.to_tensor(_ndarr=_image, _dtype=self.dtype_image), self.to_tensor(_ndarr=_mask, _dtype=self.dtype_mask), self.to_tensor(_ndarr=_bbox.reshape((-1, 1)), _dtype=self.dtype_bbox)

    def to_tensor(self, _ndarr: np.ndarray, _dtype: type) -> torch.Tensor:
        return F.to_tensor(_ndarr).type(_dtype)


class Normalizes(object):
    def __init__(self, _mean: tuple, _std: tuple, _inplace: bool = False, _div: float = 255.0) -> None:
        self.mean = list(_mean)
        self.std = list(_std)
        self.inplace = _inplace
        self.div = _div

    def __call__(self, _image: torch.Tensor, _mask: torch.Tensor, _bbox: torch.Tensor) -> tuple:
        return self.run(_image=_image), _mask, _bbox

    def run(self, _image: torch.Tensor) -> torch.Tensor:
        return F.normalize(_image.div(255.0), mean=self.mean, std=self.std, inplace=self.inplace)


class ToNdarr(object):
    def __init__(self, _mul: float = 255.0, _axis: tuple = (1, 2, 0), _min: float = 0.0, _max: float = 255.0, _is_round: bool = True, _round_decimals: int = 0, _dtype: type = np.uint8) -> None:
        self.mul = _mul
        self.axis = _axis
        self.min = _min
        self.max = _max
        self.is_round = _is_round
        self.round_decimals = _round_decimals
        self.dtype = _dtype

    def __call__(self, _input: torch.Tensor) -> np.ndarray:
        return self.run(_input=_input, _mul=self.mul, _axis=self.axis, _min=self.min, _max=self.max, _is_round=self.is_round, _round_decimals=self.round_decimals, _dtype=self.dtype)

    @classmethod
    def run(self, _input: torch.Tensor, _mul: float = 255.0, _axis: tuple = (1, 2, 0), _min: float = 0.0, _max: float = 255.0, _is_round: bool = True, _round_decimals: int = 0, _dtype: type = np.uint8) -> np.ndarray:
        return imgcv_.casting(_ndarr=imgcv_.clip(_ndarr=_input.mul(_mul).cpu().numpy().transpose(_axis), _min=_min, _max=_max, _is_round=_is_round, _round_decimals=_round_decimals), _dtype=_dtype)


class Denormalize(object):
    def __init__(self, _mean: tuple, _std: tuple, _inplace: bool = False) -> None:
        self.mean = list(_mean)
        self.std = list(_std)
        self.inplace = _inplace

    def __call__(self, _input: torch.Tensor) -> torch.Tensor:
        return self.run(_input=_input, _mean=self.mean, _std=self.std, _inplace=self.inplace)

    @classmethod
    def run(self, _input: torch.Tensor, _mean: list, _std: list, _inplace: bool = False) -> torch.Tensor:
        if _inplace is False:
            _input = _input.clone()

        dtype = _input.dtype
        device = _input.device
        mean = torch.as_tensor(_mean, dtype=dtype, device=device)
        std = torch.as_tensor(_std, dtype=dtype, device=device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)

        return _input.mul_(std).add_(mean)


if __name__ == '__main__':
    pass
