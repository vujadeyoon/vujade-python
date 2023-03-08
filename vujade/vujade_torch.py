"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_torch.py
Description: A module for PyTorch

Acknowledgement:
    1. This implementation is highly inspired from thstkdgus35 and 4uiiurz1.
    2. Github:
        i)  https://github.com/thstkdgus35/EDSR-PyTorch
        ii) https://github.com/4uiiurz1/pytorch-deform-conv-v2

"""


import math
import copy
import _collections
import cpuinfo
import torch
import torchvision
import torchinfo
import torch.nn as nn
from collections import OrderedDict
from typing import Optional
from ptflops import get_model_complexity_info
from vujade import vujade_datastructure as ds_
from vujade import vujade_profiler as prof_
from vujade import vujade_flops_counter as flops_counter_
from vujade import vujade_utils as utils_
from vujade.vujade_debug import printd


COMPLEXITY_TIME_ITER = 10
COMPLEXITY_TIME_WARMUP = 5


class PyTorchUtils(object):
    @staticmethod
    def convert_ckpt(_state_dict: OrderedDict, _device: torch.device = torch.device('cpu')) -> OrderedDict:
        device = torch.device(_device)
        res = OrderedDict()
        res.update((_key.replace('module.', ''), _value.to(device)) for _idx, (_key, _value) in enumerate(_state_dict.items()))
        return res

    @staticmethod
    def is_cuda_model(_model) -> bool:
        return next(_model.parameters()).is_cuda

    @staticmethod
    def prepare_device(_is_cuda: bool) -> tuple:
        if _is_cuda is True:
            device = torch.device('cuda')

            is_gpu_available = torch.cuda.is_available()
            num_gpu_available = torch.cuda.device_count()
            gpu_ids_available = list(range(num_gpu_available))

            gpu_ids_required = utils_.get_env_var(_name_var='CUDA_VISIBLE_DEVICES', _default='',
                                                  _is_raise_existed=True).split(',')
            num_gpu_required = 0 if gpu_ids_required[0] == '' else len(gpu_ids_required)
            is_gpu_required = False if num_gpu_required == 0 else True
            gpu_ids_required = list(map(int, gpu_ids_required)) if is_gpu_required is True else list()

            if not (is_gpu_available is is_gpu_required):
                raise ValueError(
                    'The is_gpu_vailable (i.e. {}) should be equal to the is_gpu_required (i.e. {}).'.format(
                        is_gpu_available, is_gpu_required))

            if num_gpu_available < num_gpu_required:
                raise ValueError('The num_gpu_required (i.e. {}) should not exceed num_gpu_available (i.e. {}).'.format(
                    num_gpu_required, num_gpu_available))

            if not (set(gpu_ids_required).issubset(set(gpu_ids_available))):
                raise ValueError(
                    'The gpu_ids_required (i.e. {}) should be subset of the gpu_ids_available (i.e. {}).'.format(
                        gpu_ids_required, gpu_ids_available))
        else:
            device = torch.device('cpu')
            gpu_ids_required = list()

        return device, gpu_ids_required


class FeatureExtractor(nn.Module):
    def __init__(self, _model, _layers: dict) -> None:
        super(FeatureExtractor, self).__init__()
        self.feature = torchvision.models._utils.IntermediateLayerGetter(_model, _layers)

    def forward(self, _x) -> list:
        return list(self.feature(_x).values())


class CalculateComputationalCost(object):
    # https://github.com/sovrasov/flops-counter.pytorch/issues/16
    def __init__(self, _units_macs: str = 'GMac', _units_flops: str = 'GFlop', _units_params: Optional[str] = None, _precision: int = 2) -> None:
        super(CalculateComputationalCost, self).__init__()
        self.units_macs = _units_macs
        self.units_flops = _units_flops
        self.precision = _precision
        self.units_params = _units_params
        if self.units_params == 'None':
            self.units_params = None

    def run(self, _model, _input_res: tuple, _print_per_layer_stat: bool = False, _as_strings: bool = True, _verbose: bool = False) -> tuple:
        macs, params = get_model_complexity_info(model=_model,
                                                 input_res=_input_res,
                                                 print_per_layer_stat=_print_per_layer_stat,
                                                 as_strings=False,
                                                 verbose=_verbose)
        flops =  2.0 * macs

        if _as_strings is True:
            macs = self._macs_to_string(_macs=macs, _units=self.units_macs, _precision=self.precision)
            flops = self._flops_to_string(_flops=flops, _units=self.units_flops, _precision=self.precision)
            params = self._params_to_string(_params_num=params, _units=self.units_params, _precision=self.precision)

        return macs, flops, params

    def _macs_to_string(self, _macs: float, _units: Optional[str] = 'GMac', _precision: int = 2) -> str:
        if _units is None:
            if _macs // 10 ** 9 > 0:
                return str(round(_macs / 10. ** 9, _precision)) + ' GMac'
            elif _macs // 10 ** 6 > 0:
                return str(round(_macs / 10. ** 6, _precision)) + ' MMac'
            elif _macs // 10 ** 3 > 0:
                return str(round(_macs / 10. ** 3, _precision)) + ' KMac'
            else:
                return str(_macs) + ' Mac'
        else:
            if _units == 'GMac':
                return str(round(_macs / 10. ** 9, _precision)) + ' ' + _units
            elif _units == 'MMac':
                return str(round(_macs / 10. ** 6, _precision)) + ' ' + _units
            elif _units == 'KMac':
                return str(round(_macs / 10. ** 3, _precision)) + ' ' + _units
            else:
                return str(_macs) + ' Mac'

    def _flops_to_string(self, _flops: float, _units: Optional[str] = 'GFlop', _precision: int = 2) -> str:
        if _units is None:
            if _flops // 10 ** 9 > 0:
                return str(round(_flops / 10. ** 9, _precision)) + ' GFlop'
            elif _flops // 10 ** 6 > 0:
                return str(round(_flops / 10. ** 6, _precision)) + ' MFlop'
            elif _flops // 10 ** 3 > 0:
                return str(round(_flops / 10. ** 3, _precision)) + ' KFlop'
            else:
                return str(_flops) + ' Flop'
        else:
            if _units == 'GFlop':
                return str(round(_flops / 10. ** 9, _precision)) + ' ' + _units
            elif _units == 'MFlop':
                return str(round(_flops / 10. ** 6, _precision)) + ' ' + _units
            elif _units == 'KFlop':
                return str(round(_flops / 10. ** 3, _precision)) + ' ' + _units
            else:
                return str(_flops) + ' Flop'

    def _params_to_string(self, _params_num: int, _units: Optional[str] = None, _precision: int = 2) -> str:
        if _units is None:
            if _params_num // 10 ** 6 > 0:
                return str(round(_params_num / 10 ** 6, 2)) + ' M'
            elif _params_num // 10 ** 3:
                return str(round(_params_num / 10 ** 3, 2)) + ' k'
            else:
                return str(_params_num)
        else:
            if _units == 'M':
                return str(round(_params_num / 10. ** 6, _precision)) + ' ' + _units
            elif _units == 'K':
                return str(round(_params_num / 10. ** 3, _precision)) + ' ' + _units
            else:
                return str(_params_num)


class DNNComplexity(object):
    def __init__(self, _model_cpu, _input_res: tuple) -> None:
        super(DNNComplexity, self).__init__()
        self.model_cpu = _model_cpu
        self.input_res = _input_res
        self.is_cuda = torch.cuda.is_available()
        self.device_name = torch.cuda.get_device_name(0) if self.is_cuda else cpuinfo.get_cpu_info()['brand_raw']
        self.device = torch.device('cuda:0' if self.is_cuda else 'cpu')
        self.tensor_cpu = torch.ones(1, *self.input_res, dtype=torch.float32)

        if self.is_cuda:
            self.model_gpu = copy.deepcopy(self.model_cpu).to(self.device)
            self.tensor_gpu = self.tensor_cpu.to(self.device)

    def develop(self) -> None:
        self._complexity_space()

    def show(self) -> None:
        printd('The model name: {}'.format(self.model_cpu.__class__), _is_pause=False)
        printd('The CUDA is available: {}.'.format(self.is_cuda), _is_pause=False)
        printd('The current single device is {}.'.format(self.device_name), _is_pause=False)
        self._complexity_space()
        self._complexity_time_cpu()
        if self.is_cuda:
            self._complexity_time_gpu()

    def summary(self) -> None:
        self._summary(_batch_size=1, _device='cpu', _is_summary=True)

    def _complexity_space(self) -> None:
        macs, flops, params = CalculateComputationalCost(
                    _units_macs='GMac',
                    _units_flops='GFlop',
                    _units_params=None,
                    _precision=2).run(
                    _model=self.model_cpu,
                    _input_res=self.input_res,
                    _print_per_layer_stat=False,
                    _as_strings=True,
                    _verbose=False)

        printd('{:<25} {}.'.format('Tensor shape: ', self.input_res), _is_pause=False)
        printd('{:<25} {}.'.format('Trainable parameters: ', params), _is_pause=False)
        printd('{:<25} {}.'.format('Macs: ', macs), _is_pause=False)
        printd('{:<25} {}.'.format('Flops: ', flops), _is_pause=False)

    @prof_.measure_time(_iter=COMPLEXITY_TIME_ITER, _warmup=COMPLEXITY_TIME_WARMUP)
    def _complexity_time_cpu(self) -> None:
        self.model_cpu(self.tensor_cpu)

    @prof_.measure_time(_iter=COMPLEXITY_TIME_ITER, _warmup=COMPLEXITY_TIME_WARMUP)
    def _complexity_time_gpu(self) -> None:
        self.model_gpu(self.tensor_gpu)

    def _summary(self, _batch_size: int = 1, _device: str = 'cpu', _is_summary: bool = True) -> str:
        if not _device in {'cpu', 'gpu'}:
            raise ValueError

        printd('{} Network summary.'.format(self.model_cpu.__class__.__name__), _is_pause=False)
        if _is_summary is True:
            printd(torchinfo.summary(self.model_cpu, input_size=(_batch_size, *self.input_res), device=_device, verbose=0), _is_pause=False)

        tensor_input = torch.randn([1, *self.input_res], dtype=torch.float).to(_device)
        counter = flops_counter_.add_flops_counting_methods(self.model_cpu)
        counter.eval().start_flops_count()
        counter(tensor_input)
        str_1 = 'Input image resolution: ({}, {}, {}, {})'.format(_batch_size, *self.input_res)
        str_2 = 'Trainable model parameters: {}'.format(self._count_parameters())
        str_3 = 'Flops: {}'.format(flops_counter_.flops_to_string(counter.compute_average_flops_cost()))
        printd(str_1, _is_pause=False)
        printd(str_2, _is_pause=False)
        printd(str_3, _is_pause=False)
        printd('----------------------------------------------------------------', _is_pause=False)

        return '{}; {}; {}.'.format(str_1, str_2, str_3)

    def _count_parameters(self) -> int:
        # Another option
        # return filter(lambda p: p.requires_grad, self.parameters())
        # return sum([np.prod(p.size()) for p in model_parameters])
        return sum(p.numel() for p in self.model_cpu.parameters() if p.requires_grad)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class GetFeatureMap(nn.Module):
    def __init__(self, _model, _return_layers):
        '''
        Usage:
            1) backbone = GetFeatureMap(_model=model_backbone, _return_layers={7: 7, 13: 13, 21: 21})
            2) y = backbone(_x=x)
            3) print([(k, v.shape) for k, v in y.items()])
            4) backbone.summary(_x=x, _is_print_module=False)
        '''
        super(GetFeatureMap, self).__init__()

        self.model_children = nn.Sequential(*list(_model.features.children()))
        self.return_layers = _return_layers

    def forward(self, _x):
        res = _collections.OrderedDict()
        self.return_layers_keys = ds_.queue(_init_list=list(self.return_layers.keys()))
        self.return_layers_vals = ds_.queue(_init_list=list(self.return_layers.values()))

        for idx, (name, module) in enumerate(self.model_children.named_children()):
            _x = module(_x)
            # print(idx, _x.shape) # For debug

            if idx == self.return_layers_keys.peek():
                self.return_layers_keys.dequeue()
                res[self.return_layers_vals.dequeue()] = _x

            if self.return_layers_keys.is_empty() is True:
                break

        return res

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def modify_last_layer_lr(named_params, base_lr, lr_mult_w, lr_mult_b):
    params = list()
    for name, param in named_params:
        if 'bias' in name:
            if 'FullyConnectedLayer_' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_b, 'weight_decay': 0}]
            else:
                params += [{'params':param, 'lr': base_lr * 2, 'weight_decay': 0}]
        else:
            if 'FullyConnectedLayer_' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_w}]
            else:
                params += [{'params':param, 'lr': base_lr * 1}]
    return params


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out


class DeformConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            offset_groups: int = 1
    ):
        super(DeformConv2d, self).__init__()

        self.dfconv = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.p_conv = nn.Conv2d(in_channels, 2 * offset_groups * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        offset = self.p_conv(x)
        res = self.dfconv(x, offset)

        return res


class DeformConv2dV2(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2dV2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset
