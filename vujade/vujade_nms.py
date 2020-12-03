"""
Dveloper: vujadeyoon
E-mail: sjyoon1671@gmail.com
Github: https://github.com/vujadeyoon/vujade
Date: Dec. 03, 2020.
Title: vujade_nms.py
Version: 0.1.1
Description: A module for Non-Maximum Suppression (NMS)
"""


import numpy as np
import torch
from vujade.utils.NMS.cython_nms.cy_nms import nms as nms_cy_ndarr_
from vujade.utils.NMS.python_nms.py_nms_ndarr import nms as nms_py_ndarr_
from vujade.utils.NMS.python_nms.py_nms_tensor import nms as nms_py_tensor_
from torchvision.ops import nms as nms_torchvision
from utils.box_utils import decode, decode_landm


def nms_cpu(_loc, _conf, _landms, _prior_data, _scale_boxes, _scale_landms, _scaling_ratio, _variance, _args, _nms='nms_cy_ndarr'):
    if _nms == 'nms_cy_ndarr':
        # Option 1: Cython based NMS for ndarray using the CPU by Ross Girshick
        nms = nms_cy_ndarr_
    elif _nms == 'nms_py_ndarr':
        # Option 2: Python based NMS for ndarray using the CPU by Ross Girshick
        nms = nms_py_ndarr_
    elif _nms == 'nms_torchvision':
        # Option 3: Python based NMS for PyTorch tensor using the CPU from the torchvision
        nms = nms_torchvision
    elif _nms == 'nms_py_tensor':
        # Option 4: Python based NMS for PyTorch tensor using the CPU by fmassa
        nms = nms_py_tensor_
    else:
        raise NotImplementedError

    boxes = decode(_loc.data.squeeze(0), _prior_data, _variance)
    boxes = boxes * _scale_boxes / _scaling_ratio
    landms = decode_landm(_landms.data.squeeze(0), _prior_data, _variance)
    landms = landms * _scale_landms / _scaling_ratio

    if _nms != 'nms_torchvision':
        boxes = boxes.cpu().numpy()
        landms = landms.cpu().numpy()
        scores = _conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > _args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:_args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

        if _nms != 'nms_py_tensor':
            keep_idx = nms(_dets=dets, _thres=_args.nms_threshold)
        else: # _nms == 'nms_py_tensor'
            keep_idx, _ = list(nms(_boxes=torch.from_numpy(dets), _scores=torch.from_numpy(scores), _thres=_args.nms_threshold, _top_k=_args.top_k))

        dets = dets[keep_idx, :]
        landms = landms[keep_idx]

        # keep top-K faster NMS
        dets = dets[:_args.keep_top_k, :]
        landms = landms[:_args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
    else:
        # _nms == 'nms_torchvision'
        scores = _conf.squeeze(0).data[:, 1]

        # ignore low scores
        keep_idx = scores > _args.confidence_threshold
        boxes_ = boxes[keep_idx, :].cpu()
        landms_ = landms[keep_idx, :].cpu()
        scores_ = scores[keep_idx].cpu()

        # NMS
        keep_idx = nms(boxes_, scores_, _args.nms_threshold)
        boxes_ = boxes_[keep_idx].view(-1, 4)
        landms_ = landms_[keep_idx].view(-1, 10)
        scores_ = scores_[keep_idx].view(-1, 1)

        dets = torch.cat((boxes_, scores_, landms_), dim=1).numpy()

    return dets


def nms_gpu(_loc, _conf, _landms, _prior_data, _scale_boxes, _scale_landms, _scaling_ratio, _variance, _args, _nms='nms_torchvision'):
    if _nms == 'nms_torchvision':
        nms = nms_torchvision
    else:
        raise NotImplementedError

    boxes = decode(_loc.data.squeeze(0), _prior_data, _variance)
    boxes = boxes * _scale_boxes / _scaling_ratio
    landms = decode_landm(_landms.data.squeeze(0), _prior_data, _variance)
    landms = landms * _scale_landms / _scaling_ratio
    scores = _conf.squeeze(0).data[:, 1]

    # ignore low scores
    keep_idx = scores > _args.confidence_threshold # Time bottleneck
    boxes_ = boxes[keep_idx, :]
    landms_ = landms[keep_idx, :]
    scores_ = scores[keep_idx]

    # NMS
    keep_idx = nms(boxes_, scores_, _args.nms_threshold)
    boxes_ = boxes_[keep_idx].view(-1, 4)
    landms_ = landms_[keep_idx].view(-1, 10)
    scores_ = scores_[keep_idx].view(-1, 1)

    dets = torch.cat((boxes_, scores_, landms_), dim=1).cpu().numpy()

    return dets
