"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_transforms.py
Description: A module for transformation

Acknowledgement:
    1. This implementation is highly inspired from ClementPinard.
    2. Github: https://github.com/ClementPinard/FlowNetPytorch
"""


import math
import numpy as np
import random
import cv2
import torch
# from imgaug import augmenters as iaa
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_utils as utils_


def array2batch(_ndarr: np.ndarray, _axis_expanded: int = 0, _axes_swapped: tuple = (0, 3, 1, 2)) -> np.ndarray:
    return np.transpose(np.expand_dims(_ndarr, axis=_axis_expanded), axes=_axes_swapped)


def ndarr2tensor(_ndarr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(_ndarr)


class Standardize(object):
    @staticmethod
    def forward(_ndarr: np.ndarray, _mean: list, _std: list) -> np.ndarray:
        return (_ndarr - _mean) / _std

    @staticmethod
    def inverse(_ndarr: np.ndarray, _mean: list, _std: list) -> np.ndarray:
        return (_ndarr * _std) + _mean


class Compose(object):
    # Composes several co_transforms together.
    # For example:
    # >>> co_transforms.Compose([
    # >>>     co_transforms.CenterCrop(10),
    # >>>     co_transforms.ToTensor(),
    # >>>  ])

    def __init__(self, co_transforms):
        super(Compose, self).__init__()
        self.co_transforms = co_transforms

    def __call__(self, inputs, target):
        for trans in self.co_transforms:
            inputs, target = trans(inputs, target)

        return inputs, target


class Normalizes(object):
    def __init__(self, _normalize_type='min_max', _min_max=(0.0, 255.0), _rescale_range=(-1.0, 1.0), _mean=(0.0, 0.0, 0.0), _std=(1.0, 1.0, 1.0)):
        super(Normalizes, self).__init__()
        self.ndarr_normalize = NdarrNormalize(_normalize_type=_normalize_type, _min_max=_min_max, _rescale_range=_rescale_range, _mean=_mean, _std=_std)
        self.tensor_normalize = TensorNormalize(_normalize_type=_normalize_type, _min_max=_min_max, _rescale_range=_rescale_range, _mean=_mean, _std=_std)

    def __call__(self, _inputs, _target):
        if (isinstance(_inputs, np.ndarray) and isinstance(_target, np.ndarray)) is  True:
            normalize = self.ndarr_normalize
        elif (torch.is_tensor(_inputs) and torch.is_tensor(_target)) is True:
            normalize = self.tensor_normalize
        else:
            raise NotImplementedError

        _inputs, _target = normalize(_inputs), normalize(_target)

        return _inputs, _target


class ToTensors(object):
    def __init__(self):
        super(ToTensors, self).__init__()
        self.totensor = ToTensor()

    def __call__(self, _inputs, _target):
        tensor_inputs, tensor_target = self.totensor(_inputs), self.totensor(_target)

        return tensor_inputs, tensor_target


class ToTensors_with_Normalize(object):
    def __init__(self, _normalize_type='min_max', _min_max=([0.0, 255.0], [0.0, 255.0]), _rescale_range=([-1.0, 1.0], [-1.0, 1.0]), _mean=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), _std=([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])):
        super(ToTensors_with_Normalize, self).__init__()
        self.normalize_type = _normalize_type
        self.min_max = _min_max
        self.rescale_range = _rescale_range
        self.mean = _mean
        self.std = _std

        self.totensor = ToTensor()
        self.normalize_inputs = TensorNormalize(_normalize_type=self.normalize_type, _min_max=self.min_max[0], _rescale_range=self.rescale_range[0], _mean=self.mean[0], _std=self.std[0])
        self.normalize_target = TensorNormalize(_normalize_type=self.normalize_type, _min_max=self.min_max[1], _rescale_range=self.rescale_range[1], _mean=self.mean[1], _std=self.std[1])

    def __call__(self, _inputs, _target):
        tensor_inputs, tensor_target = self.totensor(_inputs), self.totensor(_target)
        tensor_inputs, tensor_target = self.normalize_inputs(tensor_inputs), self.normalize_target(tensor_target)

        return tensor_inputs, tensor_target


class Resize(object):
    def __init__(self, _size, _interp=cv2.INTER_CUBIC):
        super(Resize, self).__init__()
        self.output_sz = _size
        self.interp = _interp

    def __call__(self, _inputs, _target):
        h0, w0, c0 = _inputs.shape

        if (self.output_sz[0] == -1):
            self.output_sz[0] = h0
        if (self.output_sz[1] == -1):
            self.output_sz[1] = w0

        if (h0 == self.output_sz[0] and w0 == self.output_sz[1]):
            return _inputs, _target

        res_intputs = cv2.resize(src=_inputs, dsize=tuple(self.output_sz), interpolation=self.interp)
        res_target = cv2.resize(src=_target, dsize=tuple(self.output_sz), interpolation=self.interp)

        return res_intputs, res_target


class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given ndarray with a probability of 0.5
    """
    def __init__(self, _prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            _inputs = _inputs[:, ::-1, :]
            _target = _target[:, ::-1, :]

        return _inputs, _target


class RandomVerticalFlip(object):
    """
    Randomly horizontally flips the given ndarray with a probability of 0.5
    """
    def __init__(self, _prob=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            _inputs = _inputs[::-1, :, :]
            _target = _target[::-1, :, :]

        return _inputs, _target


class RandomRotate90(object):
    """
    Randomly rotate the given ndarray with 90 degree num_rot-times.
    """
    def __init__(self, _prob=0.5):
        super(RandomRotate90, self).__init__()
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            num_rot = random.randint(0, 3)
            _inputs = np.rot90(_inputs, num_rot, axes=(0, 1))
            _target = np.rot90(_target, num_rot, axes=(0, 1))

        return _inputs, _target


class RandomRotate(object):
    """
    The imgaug-python-package based random rotate transform
    Usage:
        1) It is recommended to use the RandomRotate transform as the last image augmentation
           because the transform makes an empty space (black) on the edge of an image.
           In other words, use it just before the ToTensors_with_Normalize transform.
        2) vujade_transforms.Compose([
           vujade_transforms.RandomRotate(_degree=(-15, 15), _prob=0.5),
           vujade_transforms.ToTensors_with_Normalize(_norm='min_max')
           ])
    """
    def __init__(self, _degree=(-30, 30), _prob=0.5):
        super(RandomRotate, self).__init__()
        self.degree_min = _degree[0]
        self.degree_max = _degree[1]
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            degree_rotate = random.randint(self.degree_min, self.degree_max)
            iaa_trans = iaa.Rotate(rotate=degree_rotate)
            _inputs = iaa_trans(image=_inputs)
            _target = iaa_trans(image=_target)

        return _inputs, _target


class RandomShearX(object):
    """
    The imgaug-python-package based random shear transform with x-axis
    """
    def __init__(self, _shear=(-30, 30), _prob=0.5):
        super(RandomShearX, self).__init__()
        self.shear_min = _shear[0]
        self.shear_max = _shear[1]
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            degree_shear = random.randint(self.shear_min, self.shear_max)
            iaa_trans = iaa.ShearX(shear=degree_shear)
            _inputs = iaa_trans(image=_inputs)
            _target = iaa_trans(image=_target)

        return _inputs, _target


class RandomShearY(object):
    """
    The imgaug-python-package based random shear transform with y-axis
    """
    def __init__(self, _shear=(-30, 30), _prob=0.5):
        super(RandomShearY, self).__init__()
        self.shear_min = _shear[0]
        self.shear_max = _shear[1]
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            degree_shear = random.randint(self.shear_min, self.shear_max)
            iaa_trans = iaa.ShearY(shear=degree_shear)
            _inputs = iaa_trans(image=_inputs)
            _target = iaa_trans(image=_target)

        return _inputs, _target


class RandomPerspective(object):
    """
    The imgaug-python-package based random perspective transform
    """
    def __init__(self, _scale=(0.0, 0.6), _prob=0.5):
        super(RandomPerspective, self).__init__()
        self.scale_min = _scale[0]
        self.scale_max = _scale[1]
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            degree_scale = random.uniform(self.scale_min, self.scale_max)
            iaa_trans = iaa.PerspectiveTransform(scale=degree_scale, keep_size=True)
            _inputs = iaa_trans(image=_inputs)
            _target = iaa_trans(image=_target)

        return _inputs, _target


class RandomElastic(object):
    """
    The imgaug-python-package based random elastic transform
    """
    def __init__(self, _alpha=(0.0, 40.0), _sigma=(4.0, 8.0), _prob=0.5):
        super(RandomElastic, self).__init__()
        self.alpha_min = _alpha[0]
        self.alpha_max = _alpha[1]
        self.sigma_min = _sigma[0]
        self.sigma_max = _sigma[1]
        self.prob = _prob

    def __call__(self, _inputs, _target):
        if random.random() < self.prob:
            degree_alpha = random.randint(self.alpha_min, self.alpha_max)
            degree_sigma = random.randint(self.sigma_min, self.sigma_max)
            iaa_trans = iaa.ElasticTransformation(alpha=degree_alpha, sigma=degree_sigma)
            _inputs = iaa_trans(image=_inputs)
            _target = iaa_trans(image=_target)

        return _inputs, _target


class RandomSpatialShuffle(object):
    """
    The novel image augmentation that is called RandomSpatialShuffle
    """
    def __init__(self, _prob=0.5):
        super(RandomSpatialShuffle, self).__init__()
        self.prob = _prob

    def __call__(self, _inputs, _target):
        assert (_inputs.ndim==3 and _target.ndim==3)
        assert (_inputs.shape == _target.shape)

        if random.random() < self.prob:
            h, w, c = _inputs.shape
            h_center = int(math.floor(h / 2.0))
            w_center = int(math.floor(w / 2.0))

            inputs_top_left, inputs_top_right = _inputs[0:h_center, 0:w_center, :], _inputs[0:h_center, w_center:, :]
            inputs_bottom_left, inputs_bottom_right = _inputs[h_center:, 0:w_center, :], _inputs[h_center:, w_center:, :]
            target_top_left, target_top_right = _target[0:h_center, 0:w_center, :], _target[0:h_center, w_center:, :]
            target_bottom_left, target_bottom_right = _target[h_center:, 0:w_center, :], _target[h_center:, w_center:, :]

            inputs_comb = [inputs_top_left, inputs_top_right, inputs_bottom_left, inputs_bottom_right]
            target_comb = [target_top_left, target_top_right, target_bottom_left, target_bottom_right]

            inputs_target_comb = list(zip(inputs_comb, target_comb))
            random.shuffle(inputs_target_comb)
            inputs_comb, target_comb = zip(*inputs_target_comb)

            _inputs = np.vstack((np.hstack((inputs_comb[0], inputs_comb[1])), np.hstack((inputs_comb[2], inputs_comb[3]))))
            _target = np.vstack((np.hstack((target_comb[0], target_comb[1])), np.hstack((target_comb[2], target_comb[3]))))

        return _inputs, _target


class RandomCutMoireBackward(object):
    def __init__(self, _size=(32, 32), _prob=0.5):
        super(RandomCutMoireBackward, self).__init__()
        self.height_new = _size[0]
        self.width_new = _size[1]
        self.prob = _prob

    def __call__(self, _inputs, _target):
        assert (_inputs.ndim == 3 and _target.ndim == 3)
        assert (_inputs.shape == _target.shape)
        assert (self.height_new <= _inputs.shape[0] or self.width_new <= _inputs.shape[1])

        if random.random() < self.prob:
            heigth_curr = _inputs.shape[0]
            width_curr = _inputs.shape[1]

            idy = random.randint(0, (heigth_curr - self.height_new))
            idx = random.randint(0, (width_curr - self.width_new))

            _inputs[idy:idy + self.height_new, idx:idx + self.width_new, :] = _target[idy:idy+self.height_new, idx:idx+self.width_new, :]


        return _inputs, _target


class GuidedFilterForward(object):
    """
    GuidedFilterForward
    Usage:
        1) It is recommended to use the GuidedFilterForward transform as the last image augmentation
           because the image can be transformed to two layers.
           In other words, use it just before the ToTensors_with_Normalize transform.
        2) vujade_transforms.Compose([
           vujade_transforms.RandomRotate(_degree=(-15, 15), _prob=0.5),
           vujade_transforms.GuidedFilterForward(),
           vujade_transforms.ToTensors_with_Normalize(_norm='min_max')
           ])
    """
    def __init__(self, _radius=5, _eps=None, _dDepth=-1, _scale=0.01, _option='layer_detail', _apply_target=False):
        super(GuidedFilterForward, self).__init__()
        self.radius = _radius
        self.eps = _eps
        self.dDepth = _dDepth
        self.scale = _scale
        self.option = _option
        self.apply_target = _apply_target

        if self.option != 'layer_detail':
            raise NotImplementedError

    def __call__(self, _inputs, _target):
        assert (isinstance(_inputs, np.ndarray))
        assert (isinstance(_target, np.ndarray))
        assert (_inputs.ndim==3 and _target.ndim==3)
        assert (_inputs.shape == _target.shape)

        res_inputs = imgcv_._guided_filter_forward(_ndarr=_inputs, _radius=self.radius, _eps=self.eps, _dDepth=self.dDepth, _scale=self.scale, _option=self.option)
        _inputs = res_inputs[0]

        if self.apply_target is True:
            res_target = imgcv_._guided_filter_forward(_ndarr=_target, _radius=self.radius, _eps=self.eps, _dDepth=self.dDepth, _scale=self.scale, _option=self.option)
            _target = res_target[0]

        return _inputs, _target


class NdarrNormalize(object):
    def __init__(self, _normalize_type='min_max', _min_max=(0.0, 255.0), _rescale_range=(-1.0, 1.0), _mean=(0.0, 0.0, 0.0), _std=(1.0, 1.0, 1.0)):
        super(NdarrNormalize, self).__init__()
        self.normalize_type = _normalize_type
        self.min = _min_max[0]
        self.max = _min_max[1]
        self.rescale_range_low = _rescale_range[0]
        self.rescale_range_high = _rescale_range[1]
        self.mean = _mean
        self.std = _std
        self.diff_min_max_range = self.max - self.min
        self.diff_rescale_range = self.rescale_range_high - self.rescale_range_low

    def __call__(self, _ndarr):
        assert (isinstance(_ndarr, np.ndarray))

        ndims = _ndarr.ndim

        if self.normalize_type == 'min_max':
            _ndarr = self._min_max(_ndarr=_ndarr)
        elif self.normalize_type == 'min_max_rescale':
            _ndarr = (self.diff_rescale_range * (_ndarr - self.min) / self.diff_min_max_range) + self.rescale_range_low
        elif self.normalize_type == 'min_max_standardization':
            _ndarr = self._min_max(_ndarr=_ndarr)
            if ndims == 3:
                assert (_ndarr.shape[2] == len(self.mean) and _ndarr.shape[2] == len(self.std))
                for idx_channel in range(_ndarr.shape[2]):
                    _ndarr[:, :, idx_channel] = (_ndarr[:, :, idx_channel] - self.mean[idx_channel]) / self.std[idx_channel]
            elif ndims == 4:
                assert (_ndarr.shape[3] == len(self.mean) and _ndarr.shape[3] == len(self.std))
                for idx_channel in range(_ndarr.shape[3]):
                    _ndarr[:, :, :, idx_channel] = (_ndarr[:, :, :, idx_channel] - self.mean[idx_channel]) / self.std[idx_channel]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return _ndarr

    def _min_max(self, _ndarr):
        return (_ndarr - self.min) / self.diff_min_max_range


class NdarrDenormalize(object):
    def __init__(self, _normalize_type='min_max', _min_max=(0.0, 255.0), _rescale_range=(-1.0, 1.0), _mean=(0.0, 0.0, 0.0), _std=(1.0, 1.0, 1.0)):
        super(NdarrDenormalize, self).__init__()
        self.normalize_type = _normalize_type
        self.min = _min_max[0]
        self.max = _min_max[1]
        self.rescale_range_low = _rescale_range[0]
        self.rescale_range_high = _rescale_range[1]
        self.mean = _mean
        self.std = _std
        self.diff_min_max_range = self.max - self.min
        self.diff_rescale_range = self.rescale_range_high - self.rescale_range_low

    def __call__(self, _ndarr):
        assert (isinstance(_ndarr, np.ndarray))

        ndims = _ndarr.ndim

        if self.normalize_type == 'min_max':
            _ndarr = self._min_max(_ndarr=np.clip(a=_ndarr, a_min=0.0, a_max=1.0))
        elif self.normalize_type == 'min_max_rescale':
            _ndarr = (self.diff_min_max_range * (np.clip(a=_ndarr, a_min=self.rescale_range_low, a_max=self.rescale_range_high) - self.rescale_range_low) / self.diff_rescale_range) + self.min
        elif self.normalize_type == 'min_max_standardization':
            if ndims == 3:
                assert (_ndarr.shape[2] == len(self.mean) and _ndarr.shape[2] == len(self.std))
                for idx_channel in range(_ndarr.shape[2]):
                    _ndarr[:, :, idx_channel] = self._min_max(_ndarr=np.clip(a=(self.std[idx_channel] * _ndarr[:, :, idx_channel]) + self.mean[idx_channel], a_min=0.0, a_max=1.0))
            elif ndims == 4:
                assert (_ndarr.shape[3] == len(self.mean) and _ndarr.shape[3] == len(self.std))
                for idx_channel in range(_ndarr.shape[3]):
                    _ndarr[:, :, :, idx_channel] = self._min_max(_ndarr=np.clip(a=(self.std[idx_channel] * _ndarr[:, :, :, idx_channel]) + self.mean[idx_channel], a_min=0.0, a_max=1.0))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return _ndarr

    def _min_max(self, _ndarr):
        return (self.diff_min_max_range * _ndarr) + self.min


class TensorNormalize(object):
    def __init__(self, _normalize_type='min_max', _min_max=(0.0, 255.0), _rescale_range=(-1.0, 1.0), _mean=(0.0, 0.0, 0.0), _std=(1.0, 1.0, 1.0)):
        super(TensorNormalize, self).__init__()
        self.normalize_type = _normalize_type
        self.min = _min_max[0]
        self.max = _min_max[1]
        self.rescale_range_low = _rescale_range[0]
        self.rescale_range_high = _rescale_range[1]
        self.mean = _mean
        self.std = _std
        self.diff_min_max_range = self.max - self.min
        self.diff_rescale_range = self.rescale_range_high - self.rescale_range_low

    def __call__(self, _tensor):
        ndims = _tensor.ndim

        if torch.is_tensor(_tensor) and self.normalize_type == 'min_max':
            _tensor = self._min_max(_tensor=_tensor)
        elif torch.is_tensor(_tensor) and self.normalize_type == 'min_max_rescale':
            _tensor = _tensor.sub(self.min).mul(self.diff_rescale_range).div(self.diff_min_max_range).add(self.rescale_range_low)
        elif torch.is_tensor(_tensor) and self.normalize_type == 'min_max_standardization':
            _tensor = self._min_max(_tensor=_tensor)
            if ndims == 3:
                assert (_tensor.shape[0] == len(self.mean) and _tensor.shape[0] == len(self.std))
                for idx_channel in range(_tensor.shape[0]):
                    _tensor[idx_channel, :, :] = _tensor[idx_channel, :, :].sub(self.mean[idx_channel]).div(self.std[idx_channel])
            elif ndims == 4:
                assert (_tensor.shape[1] == len(self.mean) and _tensor.shape[1] == len(self.std))
                for idx_channel in range(_tensor.shape[1]):
                    _tensor[:, idx_channel, :, :] = _tensor[:, idx_channel, :, :].sub(self.mean[idx_channel]).div(self.std[idx_channel])
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return _tensor

    def _min_max(self, _tensor):
        return _tensor.sub(self.min).div(self.diff_min_max_range)



class TensorDenormalize(object):
    def __init__(self, _normalize_type='min_max', _min_max=(0.0, 255.0), _rescale_range=(-1.0, 1.0), _mean=(0.0, 0.0, 0.0), _std=(1.0, 1.0, 1.0)):
        super(TensorDenormalize, self).__init__()
        self.normalize_type = _normalize_type
        self.min = _min_max[0]
        self.max = _min_max[1]
        self.rescale_range_low = _rescale_range[0]
        self.rescale_range_high = _rescale_range[1]
        self.mean = _mean
        self.std = _std
        self.diff_min_max_range = self.max - self.min
        self.diff_rescale_range = self.rescale_range_high - self.rescale_range_low

    def __call__(self, _tensor):
        ndims = _tensor.ndim

        if torch.is_tensor(_tensor) and self.normalize_type == 'min_max':
            _tensor = self._min_max(_tensor=torch.clamp(_tensor, min=0.0, max=1.0))
        elif torch.is_tensor(_tensor) and self.normalize_type == 'min_max_rescale':
            _tensor = torch.clamp(_tensor, min=self.rescale_range_low, max=self.rescale_range_high).sub(self.rescale_range_low).mul(self.diff_min_max_range).div(self.diff_rescale_range).add(self.min)
        elif torch.is_tensor(_tensor) and self.normalize_type == 'min_max_standardization':
            if ndims == 3:
                assert (_tensor.shape[0] == len(self.mean) and _tensor.shape[0] == len(self.std))
                for idx_channel in range(_tensor.shape[0]):
                    _tensor[idx_channel, :, :] = self._min_max(_tensor=torch.clamp(_tensor[idx_channel, :, :].mul(self.std[idx_channel]).add(self.mean[idx_channel]), min=0.0, max=1.0))
            elif ndims == 4:
                assert (_tensor.shape[1] == len(self.mean) and _tensor.shape[1] == len(self.std))
                for idx_channel in range(_tensor.shape[1]):
                    _tensor[:, idx_channel, :, :] = self._min_max(_tensor=torch.clamp(_tensor[:, idx_channel, :, :].mul(self.std[idx_channel]).add(self.mean[idx_channel]), min=0.0, max=1.0))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return _tensor

    def _min_max(self, _tensor):
        return _tensor.mul(self.diff_min_max_range).add(self.min)


class ToTensor(object):
    def __call__(self, _ndarr):
        return _totensor(_ndarr=_ndarr)


class ToNdarr(object):
    def __call__(self, _tensor):
        return _tondarr(_tensor=_tensor)


def _totensor(_ndarr):
    """
    It is recommended to use ToTensor() class instead of the _totensor function.
    """
    assert (isinstance(_ndarr, np.ndarray))

    ndims = _ndarr.ndim
    if ndims == 3:
        return torch.from_numpy(_ndarr.copy().transpose(2, 0, 1))
    elif ndims == 4:
        return torch.from_numpy(_ndarr.copy().transpose(0, 3, 1, 2))
    else:
        raise NotImplementedError


def _tondarr(_tensor):
    """
    It is recommended to use ToNdarr() class instead of the _tondarr function.
    """
    ndims = _tensor.ndim
    if torch.is_tensor(_tensor) and ndims == 3:
        return _tensor.detach().cpu().numpy().transpose(1, 2, 0)
    elif torch.is_tensor(_tensor) and ndims == 4:
        return _tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError