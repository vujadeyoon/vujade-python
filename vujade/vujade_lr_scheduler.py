"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_lr_scheduler.py
Description: A module for learning rate scheduler
"""

from math import log, exp


def adjust_learning_rate(_opt, _optimizer, _epoch, _type='poly'):
    if _type == 'step':
        epoch_iter = (_epoch - 1) // _opt.lr_decay
        lr = _opt.lr / (2 ** epoch_iter)
    elif _type == 'exp':
        k = log(2) / _opt.lr_decay
        lr = _opt.lr * exp(-k * (_epoch - 1))
    elif _type == 'inv':
        k = 1 / _opt.lr_decay
        lr = _opt.lr / (1 + k * (_epoch - 1))
    elif _type == 'poly':
        power = 0.9
        lr = _opt.lr * ((1.0 - ((_epoch - 1) / _opt.nEpochs)) ** power)
    else:
        raise NotImplementedError

    for param_group in _optimizer.param_groups:
        param_group['lr'] = lr
