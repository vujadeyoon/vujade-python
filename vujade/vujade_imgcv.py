"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_imgcv.py
Description: A module for image processing and computer vision
             (Commented codes need to be checked because they may not be compatible with the current version.)
"""


import os
import numpy as np
import cv2
import cv2.ximgproc as ximgproc
import pywt
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing.queues import Queue
from typing import Optional, Set, Union
from vujade import vujade_multiprocess as multiprocess_
from vujade import vujade_path as path_
from vujade.vujade_debug import printd


class Transform(object):
    @staticmethod
    def quad(_ndarr_img: np.ndarray, _dsize: tuple, _quad_src: np.ndarray, _flags: int = cv2.INTER_LINEAR, _borderMode: int = cv2.BORDER_REFLECT, _borderValue: tuple = (0, 0, 0)) -> tuple:
        quad_dst = np.asarray([[0, 0], [0, _dsize[0] - 1], [_dsize[1] - 1, _dsize[0] - 1], [_dsize[1] - 1, 0]], dtype=np.float32)
        matrix_perspective = cv2.getPerspectiveTransform(_quad_src, quad_dst).astype(np.float32)
        return cv2.warpPerspective(
            src=_ndarr_img,
            M=matrix_perspective,
            dsize=_dsize,
            flags=_flags,
            borderMode=_borderMode,
            borderValue=_borderValue
        ), matrix_perspective

    @staticmethod
    def rotate(_ndarr_img: np.ndarray, _degree: float, _flags: int = cv2.INTER_LINEAR, _borderMode: int = cv2.BORDER_REFLECT, _borderValue: tuple = (0, 0, 0)) -> np.ndarray:
        """
        Reference:
            i)  https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#void%20warpAffine(InputArray%20src,%20OutputArray%20dst,%20InputArray%20M,%20Size%20dsize,%20int%20flags,%20int%20borderMode,%20const%20Scalar&%20borderValue)
            ii) http://kocw-n.xcache.kinxcdn.com/data/document/2021/kumoh/leehaeyeoun0824/20.pdf
        """
        ndarr_img_height, ndarr_img_width, _ = _ndarr_img.shape
        mat_affine = cv2.getRotationMatrix2D((ndarr_img_width // 2, ndarr_img_height // 2), _degree, 1.0)
        return cv2.warpAffine(
            src=_ndarr_img,
            M=mat_affine,
            dsize=(ndarr_img_width, ndarr_img_height),
            flags=_flags,
            borderMode=_borderMode,
            borderValue=_borderValue
        )

    @staticmethod
    def warp(_ndarr_pts_src: np.ndarray, _matrix: np.ndarray, _is_normalize: bool = True):
        if _ndarr_pts_src.ndim != 2:
            raise ValueError('The _ndarr_pts_src.ndim should be 2.')

        if _matrix.ndim != 2:
            raise ValueError('The _matrix.ndim should be 2.')

        ndarr_pts_3d = np.concatenate((_ndarr_pts_src, np.ones(shape=(_ndarr_pts_src.shape[0], 1), dtype=np.float32)), axis=1)
        ndarr_pts_dst = _matrix @ ndarr_pts_3d.T

        if _is_normalize is True:
            ndarr_pts_dst[0, :] /= ndarr_pts_dst[2, :]
            ndarr_pts_dst[1, :] /= ndarr_pts_dst[2, :]

        return ndarr_pts_dst

    @staticmethod
    def get_ndarr_vector(_ndarr_src: np.ndarray, _ndarr_dst: np.ndarray, _is_left_handed: bool = False):
        if (_ndarr_src.ndim != 2) or (_ndarr_dst.ndim != 2):
            raise NotImplementedError('The both _ndarr_src.ndim and _ndarr_dst.ndim should be 2.')

        res = _ndarr_dst - _ndarr_src
        if _is_left_handed is True:
            res[:, 1] *= -1

        return res

    @staticmethod
    def get_rotation_matrix_2d(_radian: float, _is_negative: bool = False) -> np.ndarray:
        res = np.asarray([[np.cos(_radian), -np.sin(_radian)], [np.sin(_radian), np.cos(_radian)]], dtype=np.float32)
        if _is_negative is True:
            res *= np.asarray([[1, -1], [-1, 1]])

        return res

    @staticmethod
    def rotate_pts_from_anchor(_ndarr_src: np.ndarray, _ndarr_anchor: np.ndarray, _ndarr_matrix_rotation: np.ndarray) -> np.ndarray:
        if (_ndarr_src.ndim != 2) or (_ndarr_anchor.ndim != 2) or (_ndarr_matrix_rotation.ndim != 2):
            raise NotImplementedError('The _ndarr_src.ndim, _ndarr_anchor.ndim and _ndarr_matrix_rotation.ndim should be 2.')
        if (_ndarr_src.shape[0] != 2) or (_ndarr_anchor.shape[0] != 2):
            raise ValueError('The both _ndarr_src.shape[0] and _ndarr_anchor.shape[0] should be 2.')

        return _ndarr_anchor + (_ndarr_matrix_rotation @ (_ndarr_src - _ndarr_anchor))


def get_color_code_bgr(_name_color: Optional[str] = None) -> Union[dict, tuple]:
    dict_color_code = {
        'red': (0, 0, 255),
        'orange': (0, 50, 255),
        'yellow': (0, 255, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'navy': (255, 5, 0),
        'purple': (255, 0, 100),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255)
    }

    if _name_color is None:
        res = dict_color_code
    else:
        if _name_color not in dict_color_code.keys():
            raise ValueError('The _name_color has not been supported yet.')
        res = dict_color_code[_name_color]

    return res


def get_img_extension() -> Set[str]:
    return {'.bmp', '.jpg', 'jpeg', '.png'}


class _ImWriterMP(multiprocess_.BaseMultiProcess):
    def __init__(self, _spath_frame_template: str, _num_proc: int = os.cpu_count(), _idx_format: str = '{:08d}') -> None:
        super(_ImWriterMP, self).__init__(_target_method=self._target_method, _num_proc=_num_proc)
        self.path_frame_template = path_.Path(_spath_frame_template)
        self.num_proc = _num_proc
        self.idx_format = _idx_format

        if self._is_valid_path_frame_template() is False:
            raise ValueError('The given file extension of the _spath_frame_template, {} should be an image extension.'.format(self.path_frame_template.str))

    def proc_enqueue(self, _list_frames: list, _list_idx_frames: list) -> None:
        # Todo: To be coded.
        for _idx, (_frame, _idx_frame) in enumerate(zip(_list_frames, _list_idx_frames)):
            path_frame_with_index = self.path_frame_template.replace_ext('_{}{}'.format(self.idx_format, self.path_frame_template.ext).format(_idx_frame))
            self.queue.put((path_frame_with_index.str, _frame))

    def _target_method(self, _queue: Queue) -> None:
        # Todo: To be coded.
        while True:
            if not _queue.empty():
                filename, ndarr = _queue.get()
                if filename is None:
                    break
                cv2.imwrite(filename=filename, img=ndarr)

    def _is_valid_path_frame_template(self) -> bool:
        if self.path_frame_template.ext in get_img_extension():
            res = True
        else:
            res = False

        return res


class ImWriterMP(_ImWriterMP):
    def __init__(self, _spath_frame_template: str, _num_proc: int = os.cpu_count(), _idx_format: str = '{:08d}') -> None:
        super(ImWriterMP, self).__init__(_spath_frame_template=_spath_frame_template, _num_proc=_num_proc, _idx_format=_idx_format)
        self._proc_setup()

    def imwrite(self, _frames: Union[list, np.ndarray], _idx_frame_start: int = 0) -> None:
        if isinstance(_frames, list):
            list_frames = _frames
        else:
            list_frames = list(_frames)
        idx_frame_end = _idx_frame_start + len(list_frames) - 1
        list_idx_frames = list(range(_idx_frame_start, idx_frame_end + 1))
        self.proc_enqueue(_list_frames=list_frames, _list_idx_frames=list_idx_frames)

    def close(self) -> None:
        self._proc_release()


class DWT2(object):
    def __init__(self, _wavelet='haar'):
        super(DWT2, self).__init__()
        self.wavelet = _wavelet

    def transform(self, _data):
        dwt2_coeffs = pywt.dwt2(data=_data, wavelet=self.wavelet) # cA, (cH, cV, cD) = dwt2_coeffs
        return dwt2_coeffs

    def inverse_transform(self, _coeffs):
        return pywt.idwt2(coeffs=_coeffs, wavelet=self.wavelet)


def check_valid_pixel(_x: int, _bit: int = 8) -> bool:
    return (0 <= _x) and (_x <= ((2 ** _bit) - 1))


def bgr2hex(_bgr: tuple) -> str:
    b, g, r = _bgr
    return ('#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)).upper()


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.tif', 'tiff'])


def imread(_filename=None, _flags=cv2.IMREAD_COLOR, _is_bgr2rgb=False):
    # Default color channel order for the OpenCV: BGRA
    img = cv2.imread(filename=_filename, flags=_flags)
    if _is_bgr2rgb is True:
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    return img


def imwrite(_filename=None, _ndarr=None, _is_rgb2bgr=False):
    # Default color channel order for the OpenCV: BGRA
    if _is_rgb2bgr is True:
        _ndarr = cv2.cvtColor(src=_ndarr, code=cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename=_filename, img=_ndarr.astype(np.uint8))


def imresize(_ndarr, _dsize, _fx=None, _fy=None, _interpolation=cv2.INTER_LINEAR):
    img = cv2.resize(src=_ndarr, dsize=_dsize, fx=_fx, fy=_fy, interpolation=_interpolation)

    return img


def imshow(_winname='Test image', _ndarr=None):
    cv2.imshow(winname=_winname, mat=_ndarr.astype(np.uint8))


def plot(_ndarr: np.ndarray, _offset: int = 12, _is_axis: bool = False, _is_bgr2rgb: bool = True) -> None:
    img_channel, img_height, img_width = _ndarr.shape
    plt.figure(figsize=(_offset, img_height / img_width * _offset))

    if _is_axis is False:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')

    if _is_bgr2rgb is True:
        res = cv2.cvtColor(src=_ndarr, code=cv2.COLOR_BGR2RGB)
    else:
        res = _ndarr

    plt.imshow(res) # Order of the channel: RGB
    plt.show()


def clip(_ndarr: np.ndarray, _min: Optional[float] = None, _max: Optional[float] = None, _is_round: bool = True, _round_decimals: int = 0) -> np.ndarray:
    if _min is None:
        _min = _ndarr.min()
    if _max is None:
        _max = _ndarr.max()
    if _is_round is True:
        _ndarr = _ndarr.round(decimals=_round_decimals)

    return np.clip(a=_ndarr, a_min=_min, a_max=_max)


def casting(_ndarr: np.ndarray, _dtype: type = np.uint8) -> np.ndarray:
    """
    It is recommended to call clip function before calling the casting function.
    """
    return _ndarr.astype(_dtype)


def bgr2ycbcr(_ndarr: np.ndarray) -> np.ndarray:
    """
    It is equal to MATLAB based rgb2ycbcr:
        img_ycbcbr_uint8 = rgb2ycbcr(img_rgb_uint8)
    C.A. Poynton, "A Technical Introduction to Digital Video", John Wiley & Sons, Inc., 1996, p. 175
    Rec. ITU-R BT.601-5, "STUDIO ENCODING PARAMETERS OF DIGITAL TELEVISION

    FOR STANDARD 4:3 AND WIDE-SCREEN 16:9 ASPECT RATIOS",
    (1982-1986-1990-1992-1994-1995), Section 3.5.

    Digital Y′CbCr (8 bits per sample) is derived from analog B'G'R' as follows:
        Y  =  16 +  (  24.996 B + 128.553 G +  65.481 R)
        Cb = 128 +  ( 112.000 B -  74.203 G -  37.797 R)
        Cr = 128 +  (- 18.214 B -  93.786 G + 112.000 R)
    """
    trans = [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]
    basis = [16.0, 128.0, 128.0]
    max_val = 255.0

    ndims = _ndarr.ndim
    if ndims==3:
        res = np.matmul(_ndarr, trans) / max_val + basis
        return res
    elif ndims==4:
        n0, h0, w0, c0 = _ndarr.shape
        res = np.zeros(shape=(n0, h0, w0, c0), dtype=np.float32)
        for idx_n in range(n0):
            res[idx_n, :, :, :] = np.matmul(_ndarr[idx_n], trans) / max_val + basis
        return res
    else:
        raise NotImplementedError


def ycbcr2bgr(_ndarr: np.ndarray):
    raise NotImplementedError


def bgr2gray(_ndarr: np.ndarray) -> np.ndarray:
    """
    It is equal to MATLAB based rgb2gray:
        img_ycbcbr_uint8 = rgb2gray(img_rgb_uint8)

    Luma coding in video systems in the WIKIPEDIA
    For images in color spaces such as Y'UV and its relatives, which are used in standard color TV and video systems
    such as PAL, SECAM, and NTSC, a nonlinear luma component (Y') is calculated directly from gamma-compressed primary
    intensities as a weighted sum, which, although not a perfect representation of the colorimetric luminance,
    can be calculated more quickly without the gamma expansion and compression used in photometric/colorimetric
    calculations. In the Y'UV and Y'IQ models used by PAL and NTSC, the rec601 luma (Y') component is computed as
        Y  = (  0.114 B + 0.587 G +  0.2989 R)
    """
    trans = [[0.114], [0.587], [0.2989]]

    ndims = _ndarr.ndim
    if ndims==3:
        res = np.matmul(_ndarr, trans)
        return res
    elif ndims==4:
        n0, h0, w0, c0 = _ndarr.shape
        res = np.zeros(shape=(n0, h0, w0, 1), dtype=np.float32)
        for idx_n in range(n0):
            res[idx_n, :, :, :] = np.matmul(_ndarr[idx_n], trans)
        return res
    else:
        raise NotImplementedError


def batch2channel(_ndarr):
    '''
    :param _ndarr: (N, H, W, C)
    :return:       (H, W, N*C)
    '''
    assert (isinstance(_ndarr, np.ndarray))
    assert (_ndarr.ndim==4)

    _ndarr_n, _ndarr_h, _ndarr_w, _ndarr_c = _ndarr.shape
    return _ndarr.transpose((1, 2, 0, 3)).reshape((_ndarr_h, _ndarr_w, -1))


def ndarr2img(_ndarr):
    """
    :param _ndarr: _ndarr.shape==(H, W, C)
    :return: pillow image
    """
    return Image.fromarray(casting(_ndarr=clip(_ndarr=_ndarr, _min=0.0, _max=255.0), _dtype=np.uint8))

def img2ndarr(_img):
    """
    :param _img: pillow image
    :return: ndarray
    """
    return np.array(_img, copy=True) # Default copy parameter: True


class Guided_Filter(object):
    def __init__(self, _radius=5, _eps=None, _dDepth=-1, _scale=0.01, _option='layer_detail'):
        """
        The class, GuidedFilter, should be used for each image because of variable, self.layer_detail_max when using the option, layer_detail_rescale.
        :param _radius: Default value: 5
        :param _eps:    Default value: None
        :param _dDepth: Default value: -1
        :param _scale:  It is used to get _eps value.
        """
        super(Guided_Filter, self).__init__()
        self.radius = _radius
        self.eps = _eps
        self.dDepth = _dDepth
        self.scale = _scale
        self.option = _option

    def forward(self, _ndarr):
        """
        :param _ndarr: image, _ndarr.dtype=np.float32, _ndarr.shape==(H, W, 3) where, channel order is cv2.COLOR_RGB
        :return:  res: res.dtype=np.float32, res.shape==(H, W, 6) where, i) layer_base, ii) layer_detail
        """
        res = _guided_filter_forward(_ndarr=_ndarr, _radius=self.radius, _eps=self.eps, _dDepth=self.dDepth, _scale=self.scale, _option=self.option)
        self.layer_detail_max = res[1]

        return res[0]

    def inverse(self, _ndarr):
        """
        :param _ndarr: _ndarr.dtype=np.float32, _ndarr.shape==(H, W,  6) where, i) layer_base, ii) layer_detail
        :return: image, res.dtype=np.float32, res.shape==(H, W, 3)
        """

        return _guided_filter_inverse(_ndarr=_ndarr, _layer_detail_max=self.layer_detail_max, _option=self.option)


def _guided_filter_forward(_ndarr, _radius=5, _eps=None, _dDepth=-1, _scale=0.01, _option='layer_detail'):
    """
    :param _ndarr: image, _ndarr.dtype=np.float32, _ndarr.shape==(H, W, 3) where, channel order is cv2.COLOR_RGB
    :return:  res: res.dtype=np.float32, res.shape==(H, W, 6) where, i) layer_base, ii) layer_detail
    """
    eps_min = 1e-7

    ndarr_guide = cv2.cvtColor(_ndarr, cv2.COLOR_RGB2GRAY)


    if _eps is None:
        _eps = ((_scale * (ndarr_guide.max() - ndarr_guide.min())) ** 2)
    if _eps == 0:
        _eps = ((_scale * (255.0 - 0.0)) ** 2)

    layer_base = ximgproc.guidedFilter(guide=ndarr_guide, src=_ndarr, radius=_radius, eps=_eps, dDepth=_dDepth)

    if _option == 'layer_detail':
        layer_detail = _ndarr / (layer_base + eps_min)
        layer_detail_max = None
    elif _option == 'layer_detail_rescale':
        layer_detail = _ndarr / (layer_base + eps_min)
        layer_detail_max = layer_detail.max()
        layer_detail = (layer_detail / (layer_detail_max + eps_min))
    else:
        raise NotImplementedError

    res_layer = np.concatenate((layer_base, layer_detail), axis=2)

    return [res_layer, layer_detail_max]


def _guided_filter_inverse(_ndarr, _layer_detail_max=None, _option='layer_detail'):
    """
    :param _ndarr: _ndarr.dtype=np.float32, _ndarr.shape==(H, W,  6) where, i) layer_base, ii) layer_detail
    :return: image, res.dtype=np.float32, res.shape==(H, W, 3)
    """
    eps_min = 1e-7

    layer_base = _ndarr[:, :, 0:3]
    layer_detail = _ndarr[:, :, 3:6]

    if _option == 'layer_detail':
        res = (layer_detail * (layer_base + eps_min))
    elif _option == 'layer_detail_rescale':
        res = ((layer_detail * (_layer_detail_max + eps_min)) * (layer_base + eps_min))
    else:
        raise NotImplementedError

    return res


# To be removed
# def guided_filter_color_4d(_tensor_batch, _radius=5, _eps=None, _eps_val=1e-7, _device=None):
#     tondarr = trans_.ToNdarr()
#     totensor = trans_.ToTensor()
#
#     ndarr_batch = tondarr(_tensor=_tensor_batch)
#
#     ndarr_img_N, ndarr_img_H, ndarr_img_W, ndarr_img_C = ndarr_batch.shape
#     ndarr_res = np.zeros((ndarr_img_N, ndarr_img_H, ndarr_img_W, 2 * ndarr_img_C), dtype=np.float32)
#
#     for idx_n in range(ndarr_img_N):
#         ndarr_res[idx_n, :, :, :] = guided_filter_color(_ndarr_img=ndarr_batch[idx_n, :, :, :], _radius=_radius, _eps=_eps, _eps_val=_eps_val)
#
#     return totensor(_ndarr=ndarr_res).to(_device)
#
#
# To be removed
# def guided_filter_color(_ndarr_img, _radius=5, _eps=None, _eps_val=1e-7):
#     '''
#     Installation: git clone https://github.com/lisabug/guided-filter.git ./external_library/guided_filter
#
#     :param _ndarr_img:
#                1) _ndarr_img.shape==(H, W, 3)
#                2) _ndarr_img.dtype==np.float32
#                3) _ndarr_img's range: [0, 1]
#     :param _radius: default value, _ratius==5
#     :param _eps: default value, None (i.e. 0.01 * ((255 * (_ndarr_img.max() - _ndarr_img.min())) ** 2))
#     :return: res:
#                1) res.shape==(H, W, 6)
#                2) res.dtype==np.float32
#                3) res's range: [0, inf)
#                4) res[:, :, 0:3]: layer_base, res[:, :, 3:6]: layer_detail
#                5) How to re-scale the range of the layer_detail to [0, 1]: layer_detail /= detail_layer.max()
#                6) How to re-construct the image from the base and detail layers: img_recon = layer_detail * layer_base
#     '''
#
#     if _eps is None:
#         _eps = 0.01 * ((255.0 * (_ndarr_img.max() - _ndarr_img.min())) ** 2)
#
#     GF = GuidedFilter(_ndarr_img, radius=_radius, eps=_eps)
#
#     layer_base = GF.filter(p=_ndarr_img)
#     layer_detail = _ndarr_img / (layer_base + _eps_val)
#
#     res = np.concatenate((layer_base, layer_detail), axis=2)
#
#     return res


# def imclamp(_list_imgs=None):
#     res = []
#     for idx, __list_imgs in enumerate(_list_imgs):
#         res.append(__list_imgs.astype(np.uint8))
#
#     return res


# def imshow(_imgs, _rows=1, titles=None, _cmap=['brg']):
#     """
#     https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
#
#     Display a list of _imgs in a single figure with matplotlib.
#
#     Parameters
#     ---------
#     _imgs: List of np.arrays compatible with plt.imshow.
#
#     _rows (Default = 1): Number of columns in figure (number of rows is
#                         set to np.ceil(n_images/float(_rows))).
#
#     titles: List of titles corresponding to each image. Must have
#             the same length as titles.
#     """
#
#     assert ((titles is None) or (len(_imgs) == len(titles)))
#     n_images = len(_imgs)
#     if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
#     fig = plt.figure()
#     for n, (image, title) in enumerate(zip(_imgs, titles)):
#         a = fig.add_subplot(_rows, np.ceil(n_images / float(_rows)), n + 1)
#         if image.ndim == 2:
#             plt.gray()
#         plt.imshow(image)
#         a.set_title(title)
#     fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
#     plt.show()


# def export_imgs(_tensor_inputs, _tensor_target, _tensor_prediction, _iter, _norm='div', _phase='Validation'):
#     in_length = int(_tensor_inputs.size(1)/3)
#
#     for idx_batch, (inputs, target, pred) in enumerate(zip(_tensor_inputs, _tensor_target, _tensor_prediction)):
#         name_imgs = 'img_{}_{}_'.format(_iter, idx_batch)
#         path_img_target = os.path.join(os.getcwd(), 'results', _phase, (name_imgs + 'target' + '.png'))
#         path_img_pred = os.path.join(os.getcwd(), 'results', _phase, (name_imgs + 'prediction' + '.png'))
#
#         if _norm == 'div':
#             ndarr_img_target = 255 * (np.transpose(target.cpu().numpy(), (1, 2, 0)))
#             ndarr_img_pred = 255 * (np.transpose(pred.cpu().numpy(), (1, 2, 0)))
#             for idx_in in range(in_length):
#                 path_img_inputs = os.path.join(os.getcwd(), 'results', _phase, (name_imgs, 'inputs_{}'.format(idx_in) + '.png'))
#                 ndarr_img_inputs = 255 * (np.transpose(inputs[(3 * idx_in):(3 * idx_in + 3), :, :].cpu().numpy(), (1, 2, 0)))
#                 imwrite(_path_img=path_img_inputs, _ndarr=ndarr_img_inputs)
#
#         elif _norm=='simple':
#             ndarr_img_target = 255 * (np.transpose(target.cpu().numpy(), (1, 2, 0)) + 1) / 2
#             ndarr_img_pred = 255 * (np.transpose(pred.cpu().numpy(), (1, 2, 0)) + 1) / 2
#             for idx_in in range(in_length):
#                 path_img_inputs = os.path.join(os.getcwd(), 'results', _phase, (name_imgs, 'inputs_{}'.format(idx_in) + '.png'))
#                 ndarr_img_inputs = 255 * (np.transpose(inputs[(3 * idx_in):(3 * idx_in + 3), :, :].cpu().numpy(), (1, 2, 0)) + 1) / 2
#                 imwrite(_path_img=path_img_inputs, _ndarr=ndarr_img_inputs)
#         else:
#             raise ValueError('The parameter, norm, may be not corrected')
#
#         imwrite(_path_img=path_img_target, _ndarr=ndarr_img_target)
#         imwrite(_path_img=path_img_pred, _ndarr=ndarr_img_pred)


# def load_img(filepath, mode='RGB'):
#     if mode == 'Y':
#         img = Image.open(filepath).convert('YCbCr')
#         pil, _, _ = img.split()
#     else:
#         pil = Image.open(filepath).convert(mode)
#
#     return pil


# def save_img(pil, filepath):
#     # res = save_img(pil=PIL.image, filepath='pillow.png')
#     pil.save(filepath)
#     return 0


# def to_pil(pic_tensor):
#     # Convert a tensor to PIL Image.
#     # Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
#     # H x W x C to a PIL.Image while preserving the value range.
#
#     # Args:
#     #     pic_tensor (Tensor or numpy.ndarray): Image to be converted to PIL.Image.
#     #
#     # Returns:
#     #     PIL.Image: Image converted to PIL.Image.
#
#     npimg = pic_tensor
#     mode = None
#
#     if isinstance(pic_tensor, torch.FloatTensor):
#         pic_tensor = pic_tensor.mul(255).byte()
#     if torch.is_tensor(pic_tensor):
#         npimg = np.transpose(pic_tensor.numpy(), (1, 2, 0))
#     assert isinstance(npimg, np.ndarray), 'pic_tensor should be Tensor or ndarray'
#
#     if npimg.shape[2] == 1:
#         npimg = npimg[:, :, 0]
#         if npimg.dtype == np.uint8:
#             mode = 'L'
#         if npimg.dtype == np.int16:
#             mode = 'I;16'
#         if npimg.dtype == np.int32:
#             mode = 'I'
#         elif npimg.dtype == np.float32:
#             mode = 'F'
#     else:
#         if npimg.dtype == np.uint8:
#             mode = 'RGB'
#     assert mode is not None, '{} is not supported'.format(npimg.dtype)
#
#     return Image.fromarray(npimg, mode=mode)


# def valid_length_for_crop(length, crop_size):
#     return length - (length % crop_size)


# def conv2d(inputs, filters, strides=1, paddings=0):
#     # Default stride is set to 1.
#     # inputs and filter should be 4D tensor, [N, C, H, W].
#     # inputs.size[1] (C1) should be equal to filters.size[1] (C2)
#     # If inputs.size(): torch.Size([N1, C1, H1, W1]) and
#     #    filters.size(): torch.Size([N2, C2, H2, W2]), then
#     #    outputs.size(): torch.Size([N1, N2, ((H1 + 2*paddings - H2)/stride) + 1, ((W1 + 2*paddings - W2)/stride) + 1])
#     # For example,
#     # If inputs.size(): torch.Size([6, 4, 7, 7]) and
#     #    filters.size(): torch.Size([2, 4, 3, 3]), then
#     #    outputs.size(): torch.Size([6, 2, 7, 7])
#
#     return F.conv2d(input=inputs, weight=filters, stride=strides, padding=paddings)


# def dense_opticalflow(prev_frame, next_frame):
#     # cv2.imshow(winname='frame2', mat=ndarray)
#     # cv2.waitKey(10000) & 0xff
#     # cv2.imwrite(filename='Title',img=ndarray)
#     #
#     # dir_video = os.path.join(os.getcwd(), 'vtest.avi')
#     # vid = skvideo.io.vread(dir_video)
#     # frame1 = vid[0]
#     # frame2 = vid[1]
#     # of_bgr, of_mag, of_ang = dense_opticalflow(prev_frame=frame1, next_frame=frame2)
#
#     prev_frame = prev_frame.astype(np.uint8)
#     next_frame = next_frame.astype(np.uint8)
#
#     prev = cv2.cvtColor(src=prev_frame, code=cv2.COLOR_BGR2GRAY)
#     next = cv2.cvtColor(src=next_frame, code=cv2.COLOR_BGR2GRAY)
#
#     opticalflow_hsv = np.zeros_like(prev_frame)
#     opticalflow_hsv[..., 1] = 255.0
#
#     flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#
#     opticalflow_hsv[..., 0] = ang * 180 / np.pi / 2
#     opticalflow_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     res = cv2.cvtColor(src=opticalflow_hsv, code=cv2.COLOR_HSV2BGR)
#
#     return res, mag, ang


# def imreisze(input_arr, output_sz, interp='bilinear', opt_dir=0):
#     if opt_dir==0:
#         h0, w0, c0 = input_arr.shape
#         res_tensor = np.zeros([output_sz[0], output_sz[1], c0])
#     else:
#         c0, h0, w0 = input_arr.shape
#         res_tensor = np.zeros([c0, output_sz[0], output_sz[1]])
#
#     if (h0 == output_sz[0] and w0 == output_sz[1]):
#         res_tensor = input_arr
#     else:
#         if opt_dir==0:
#             for idx_c in range(c0):
#                 res_tensor[:, :, idx_c] = misc.imresize(arr=input_arr[:, :, idx_c], size=output_sz, interp=interp)
#         else:
#             for idx_c in range(c0):
#                 res_tensor[idx_c, :, :] = misc.imresize(arr=input_arr[idx_c, :, :], size=output_sz, interp=interp)
#
#     return res_tensor


# def _im2col(input_data, kernel_h, kernel_w, stride=1, pad=0):
#     """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
#
#     Parameters
#     ----------
#     input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
#     kernel_h : 필터의 높이
#     kernel_w : 필터의 너비
#     stride : 스트라이드
#     pad : 패딩
#
#     Returns
#     -------
#     col : 2차원 배열
#     """
#     N, C, H, W = input_data.shape
#     out_h = (H + 2 * pad - kernel_h) // stride + 1
#     out_w = (W + 2 * pad - kernel_w) // stride + 1
#
#     img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
#     col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
#
#     for y in range(kernel_h):
#         y_max = y + stride * out_h
#         for x in range(kernel_w):
#             x_max = x + stride * out_w
#             col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
#
#     col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
#     return col
#
#
# def col2im(col, input_shape, kernel_h, kernel_w, stride=1, pad=0):
#     """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
#
#     Parameters
#     ----------
#     col : 2차원 배열(입력 데이터)
#     input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
#     kernel_h : 필터의 높이
#     kernel_w : 필터의 너비
#     stride : 스트라이드
#     pad : 패딩
#
#     Returns
#     -------
#     img : 변환된 이미지들
#     """
#     N, C, H, W = input_shape
#     out_h = (H + 2 * pad - kernel_h) // stride + 1
#     out_w = (W + 2 * pad - kernel_w) // stride + 1
#     col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
#
#     img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
#     for y in range(kernel_h):
#         y_max = y + stride * out_h
#         for x in range(kernel_w):
#             x_max = x + stride * out_w
#             img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
#
#     return img[:, :, pad:H + pad, pad:W + pad]


# def im2col(tensor_in, kernel_size, stride=1, pad=0, tensor_out=True, _device=torch.device('cpu')):
#     kernel_h, kernel_w = kernel_size
#
#     if torch.is_tensor(tensor_in):
#         ndarr_in = tensor_in.cpu().numpy()
#     else:
#         ndarr_in = tensor_in
#
#     N, C, H, W = ndarr_in.shape
#     out_h = (H + 2 * pad - kernel_h) // stride + 1
#     out_w = (W + 2 * pad - kernel_w) // stride + 1
#
#     img = np.pad(ndarr_in, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
#     col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))
#
#     for y in range(kernel_h):
#         y_max = y + stride * out_h
#         for x in range(kernel_w):
#             x_max = x + stride * out_w
#             col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
#
#     ndarr_out = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
#
#     if tensor_out==True:
#         res = torch.from_numpy(ndarr_out).view(N, C, -1, kernel_h * kernel_w).to(_device)
#     else:
#         res = ndarr_out
#
#     return res
