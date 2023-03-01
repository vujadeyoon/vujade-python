"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_metric.py
Description: A module to measure performance for a developed DNN model

Acknowledgement:
    1. This implementation is highly inspired from HolmesShuan.
    2. Github: https://github.com/HolmesShuan/EDSR-ssim
"""


import math
import numpy as np
from scipy import signal
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_utils as utils_
import skimage
if utils_.get_pkg_version(_pkg_version=skimage.__version__)>=161:
    from skimage.metrics import structural_similarity as SSIM
else:
    from skimage.measure import compare_ssim as SSIM



class BaseMetricMeter(object):
    def __init__(self):
        super(BaseMetricMeter, self).__init__()
        self.initialized = False
        self.val = None
        self.avg = None
        self.cumsum = None
        self.count = None

    def update(self, val, weight=1.0):
        if self.initialized is False:
            self._initialize(val, weight)
        else:
            self._add(val, weight)

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def _initialize(self, val, weight):
        self.val = [val]
        self.avg = val
        self.cumsum = (val * weight)
        self.count = weight
        self.initialized = True

    def _add(self, val, weight):
        self.val.append(val)
        self.cumsum += (val * weight)
        self.count += weight
        self.avg = self.cumsum / self.count


class MetricMeter(BaseMetricMeter):
    def __init__(self, _batch_size=1):
        super(MetricMeter, self).__init__()
        self.batch_size = _batch_size
        self.loss = BaseMetricMeter()
        self.mse = BaseMetricMeter()
        self.psnr = BaseMetricMeter()
        self.ssim = BaseMetricMeter()

    def calculate(self, _loss, _ndarr_input, _ndarr_ref):
        ndarr_input = imgcv_.batch2channel(_ndarr=_ndarr_input)
        ndarr_ref = imgcv_.batch2channel(_ndarr=_ndarr_ref)
        mse_batch_val = mse_batch(_ndarr_input=ndarr_input, _ndarr_ref=ndarr_ref, _num_colr_channel=3)

        self.loss.update(_loss)
        self.mse.update(val=(mse_batch_val.sum() / self.batch_size))
        self.psnr.update(val=((psnr_batch(_mse_batch_val=mse_batch_val)).sum() / self.batch_size))
        self.ssim.update(val=ssim_skimage(_ndarr_input=ndarr_input, _ndarr_ref=ndarr_ref, _multichannel=True))


def mse_batch(_ndarr_input, _ndarr_ref, _num_colr_channel=3):
    """
    :param _ndarr_input: _ndarr_input's range: (0, 255), dtype: np.float32
    :param _ndarr_ref:   _ndarr_ref's   range: (0, 255), dtype: np.float32
    :return:             ndarray
    Usage:
        1) The Bath is deal with channel.
           Thus, it is recommended to call batch2channel function before the mse_batch function.
        2) cumsum_mse_rgb += (metric_.mse_batch(_ndarr_input=imgcv_.batch2channel(_ndarr=ndarr_input),
                              _ndarr_ref=imgcv_.batch2channel(_ndarr=ndarr_ref), _num_colr_channel=3)).sum()

    """
    if _ndarr_input.dtype == 'uint8' or _ndarr_ref.dtype == 'uint8':
        raise ValueError('The ndarray.dtype should not be uint8.')

    suqae_error = np.mean(((_ndarr_input - _ndarr_ref) ** 2), axis=(0, 1))
    return np.mean(suqae_error.reshape(-1, _num_colr_channel), axis=1)


def psnr_batch(_mse_batch_val):
    """
    :param _mse_val_each: ndarray
    :return:              ndarray
    Usage:
        1) The Bath is deal with channel.
           Thus, it is recommended to call mse_batch function before the psnr_batch function.
        2) cumsum_psnr_rgb += (metric_.psnr_batch(_mse_batch_val=(metric_.mse_batch(_ndarr_input=imgcv_.batch2channel(_ndarr=ndarr_input),
                               _ndarr_ref=imgcv_.batch2channel(_ndarr=ndarr_ref), _num_colr_channel=3)))).sum()
    """
    return (10 * np.log10((255.0 ** 2) / _mse_batch_val))


def mse(_ndarr_input, _ndarr_ref):
    """
    :param _ndarr_input: _ndarr_input's range: (0, 255), dtype: np.float32
    :param _ndarr_ref:   _ndarr_ref's   range: (0, 255), dtype: np.float32
    :return:             MSE value
    """
    if _ndarr_input.dtype == 'uint8' or _ndarr_ref.dtype == 'uint8':
        raise ValueError('The ndarray.dtype should not be uint8.')

    return ((_ndarr_input - _ndarr_ref) ** 2).mean()


def psnr(_mse_val):
    return (10 * math.log10((255.0 ** 2) / _mse_val))


def ssim_skimage(_ndarr_input, _ndarr_ref, _multichannel=False, _win_size=11, _K1=0.01, _K2=0.03, _sigma=1.5, _R=255.0):
    """
    :param _ndarr_input:  _ndarr_input's range: (0, 255), dtype: np.float32
    :param _ndarr_ref:    _ndarr_ref's range:   (0, 255), dtype: np.float32
    :param _multichannel: True of False
    :return:              SSIM value
    """
    if _ndarr_input.dtype == 'uint8' or _ndarr_ref.dtype == 'uint8':
        raise ValueError('The ndarray.dtype should not be uint8.')

    if (3<=_ndarr_input.ndim and 2<=_ndarr_input.shape[2]) and (3<=_ndarr_ref.ndim and 2<=_ndarr_ref.shape[2]):
        _multichannel=True
    elif _ndarr_input.ndim==2 and _ndarr_ref.ndim==2:
        _multichannel=False

    return SSIM(_ndarr_input/_R, _ndarr_ref/_R, multichannel=_multichannel, win_size=_win_size, data_range=1.0, gaussian_weights=True, K1=_K1, K2=_K2, sigma=_sigma)



def ssim_matlab(_ndarr_input, _ndarr_ref, _multichannel=False, _win_size=11, _K1=0.01, _K2=0.03, _sigma=1.5, _R=255.0):
    """
    :param _ndarr_input:  _ndarr_input's range: (0, 255), dtype: np.float32
    :param _ndarr_ref:    _ndarr_ref's range:   (0, 255), dtype: np.float32
    :param _multichannel: True of False
    :param _win_size:     11
    :param _K1:           0.01
    :param _K2:           0.03
    :param _sigma:        1.5
    :param _R:            255.0
    :return:              SSIM value

    _ndarr_input : y channel (i.e., luminance) of transformed YCbCr space of _ndarr_input
    _ndarr_ref : y channel (i.e., luminance) of transformed YCbCr space of _ndarr_ref
    Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
    Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
    The authors of EDSR use MATLAB's ssim as the evaluation tool,
    thus this function is the same as the ssim.m in MATLAB with C(3) == C(2)/2.
    """

    if _ndarr_input.dtype == 'uint8' or _ndarr_ref.dtype == 'uint8':
        raise ValueError('The ndarray.dtype should not be uint8.')

    if (3<=_ndarr_input.ndim and 2<=_ndarr_input.shape[2]) and (3<=_ndarr_ref.ndim and 2<=_ndarr_ref.shape[2]):
        _multichannel=True
    elif _ndarr_input.ndim==2 and _ndarr_ref.ndim==2:
        _multichannel=False

    if _multichannel is True:
        utils_.print_color(_str='[WARNING]: It is recommend to use ssim_skimage instead of ssim_matlab for the multichannel option because of computational time.', _color='WARNING')
        cumsum_ssim = 0.0
        for idx_channel in range(_ndarr_input.shape[2]):
            cumsum_ssim += _ssim_matlab(_ndarr_input=_ndarr_input[:, :, idx_channel], _ndarr_ref=_ndarr_ref[:, :, idx_channel], _win_size=_win_size, _K1=_K1, _K2=_K2, _sigma=_sigma, _R=_R)
        ssim = (cumsum_ssim / _ndarr_input.shape[2])
    else:
        ssim = _ssim_matlab(_ndarr_input=_ndarr_input, _ndarr_ref=_ndarr_ref, _win_size=_win_size, _K1=_K1, _K2=_K2, _sigma=_sigma, _R=_R)

    return ssim


def _ssim_matlab(_ndarr_input, _ndarr_ref, _win_size=11, _K1=0.01, _K2=0.03, _sigma=1.5, _R=255.0):
    gaussian_filter = _matlab_style_gauss2D(_shape=(_win_size, _win_size), _sigma=_sigma)

    X = _ndarr_input.astype(np.float64)
    Y = _ndarr_ref.astype(np.float64)

    window = gaussian_filter

    ux = signal.convolve2d(X, window, mode='same', boundary='symm')
    uy = signal.convolve2d(Y, window, mode='same', boundary='symm')

    uxx = signal.convolve2d(X * X, window, mode='same', boundary='symm')
    uyy = signal.convolve2d(Y * Y, window, mode='same', boundary='symm')
    uxy = signal.convolve2d(X * Y, window, mode='same', boundary='symm')

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    C1 = (_K1 * _R) ** 2
    C2 = (_K2 * _R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    mssim = S.mean()

    return mssim


def _matlab_style_gauss2D(_shape=(3, 3), _sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
    """

    m, n = [(ss - 1.) / 2. for ss in _shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * _sigma * _sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    return h


def get_iou(_ndarr_bbox_1, _ndarr_bbox_2):
    # _ndarr_bbox = (x1, y1, x2, y2)
    box1_area = (_ndarr_bbox_1[2] - _ndarr_bbox_1[0] + 1) * (_ndarr_bbox_1[3] - _ndarr_bbox_1[1] + 1)
    box2_area = (_ndarr_bbox_2[2] - _ndarr_bbox_2[0] + 1) * (_ndarr_bbox_2[3] - _ndarr_bbox_2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(_ndarr_bbox_1[0], _ndarr_bbox_2[0])
    y1 = max(_ndarr_bbox_1[1], _ndarr_bbox_2[1])
    x2 = min(_ndarr_bbox_1[2], _ndarr_bbox_2[2])
    y2 = min(_ndarr_bbox_1[3], _ndarr_bbox_2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)

    return iou
