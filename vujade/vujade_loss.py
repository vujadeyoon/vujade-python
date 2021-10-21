"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_loss.py
Description: A module for loss
"""


import numpy as np
import torch
import torch.nn.functional as F
from vujade import vujade_transforms as trans_


class CEloss(object):
    def __init__(self, _ignore_index=-1):
        super(CEloss, self).__init__()
        self.ignore_index = _ignore_index

    def __call__(self, _logits, _targets):
        return _celoss(_logits=_logits, _targets=_targets, _ignore_index=self.ignore_index)


def _celoss(_logits, _targets, _ignore_index):
    """
    :param _logits:  _logits.dtype==torch.float32, _logits.shape==(N, C, H, W)
    :param _targets: _targets.dtype==torch.int32,  _targets.shape==(N, H, W),  _targets's value interval: (0, 1, ..., C-1)
    :return: corss entropy loss
    """

    return F.cross_entropy(input=_logits, target=_targets.type(torch.int64), ignore_index=_ignore_index)


class L1loss_charbonnier(torch.nn.Module):
    def __init__(self, _eps=1e-3):
        super(L1loss_charbonnier, self).__init__()
        self.eps = _eps

    def forward(self, _input, _ref):
        return _l1loss_charbonnier(_input=_input, _ref=_ref, _eps=self.eps)


def _l1loss_charbonnier(_input, _ref, _eps=1e-3):
    return torch.mean(torch.sqrt(((_input - _ref) ** 2) + (_eps ** 2)))


class L1loss_sobel(object):
    def __init__(self, _device):
        super(L1loss_sobel, self).__init__()
        self.device = _device

        sobel_kernel_x = np.expand_dims(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32'), axis=2)
        sobel_kernel_y = np.expand_dims(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32'), axis=2)
        sobel_kernel_x = np.concatenate((sobel_kernel_x, sobel_kernel_x, sobel_kernel_x), axis=2)
        sobel_kernel_y = np.concatenate((sobel_kernel_y, sobel_kernel_y, sobel_kernel_y), axis=2)


        self.sobel_kernel_x = trans_._tondarr(_tensor=sobel_kernel_x.reshape((1, 3, 3, 3))).to(self.device)
        self.sobel_kernel_y = trans_._tondarr(_tensor=sobel_kernel_y.reshape((1, 3, 3, 3))).to(self.device)

    def forward(self, _input, _ref):
        input_grad_x = F.conv2d(_input, self.sobel_kernel_x, bias=None, stride=1, padding=0, dilation=1, groups=1)
        input_grad_y = F.conv2d(_input, self.sobel_kernel_y, bias=None, stride=1, padding=0, dilation=1, groups=1)
        input_grad = torch.abs(input_grad_x) + torch.abs(input_grad_y)

        ref_grad_x = F.conv2d(_ref, self.sobel_kernel_x, bias=None, stride=1, padding=0, dilation=1, groups=1)
        ref_grad_y = F.conv2d(_ref, self.sobel_kernel_y, bias=None, stride=1, padding=0, dilation=1, groups=1)
        ref_grad = torch.abs(ref_grad_x) + torch.abs(ref_grad_y)

        loss = torch.mean(torch.sqrt(input_grad - ref_grad))

        return loss