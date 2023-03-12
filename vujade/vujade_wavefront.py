"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_obj.py
Description: A module for wavefront
"""


import numpy as np
from typing import List, Optional, Set, Union
from vujade import vujade_path as path_
from vujade import vujade_text as text_
from vujade.vujade_debug import printf


class Wavefront(object):
    def __init__(self, _spath_obj: str) -> None:
        super(Wavefront, self).__init__()
        self.path_obj = path_.Path(_spath_obj)
        self.data_parsed = self._parse(self._read_obj())
        self.num_digit = {
            'v': 6,
            'vt': 6,
            'vn': 4
        }

    def get_volume_from_triangle_faces(self) -> float:
        triangle_volumes = list()
        faces = self.data_parsed['f']

        for face in faces:
            vertex_1 = np.asarray(self.data_parsed['v'][face['v'][0]])
            vertex_2 = np.asarray(self.data_parsed['v'][face['v'][1]])
            vertex_3 = np.asarray(self.data_parsed['v'][face['v'][2]])
            volume = np.dot(vertex_1, np.cross(vertex_2, vertex_3)) / 6.0
            triangle_volumes.append(volume)

        res = abs(sum(triangle_volumes))

        return res

    def setattr_v(self, _v: List[float]) -> None:
        if len(_v) != len(self.data_parsed['v']):
            raise ValueError("Both len(_v) and len(self.data_parsed['v']) should be same.")

        self.data_parsed['v'] = _v

    def write_obj(self, _spath_obj: str, _attributes_append: Optional[List[str]] = None, _attributes_written: Union[Set[str], str] = 'all', _num_digit: Optional[dict] = None) -> None:
        if _num_digit is None:
            num_digit = self.num_digit
        else:
            num_digit = _num_digit

        text_obj = text_.TEXT(_spath_filename=_spath_obj, _mode='w')

        if _attributes_append is None:
            temp = list()
        else:
            temp = _attributes_append

        for _idx, (_attr, _values) in enumerate(self.data_parsed.items()):
            for _idy, _value in enumerate(_values):
                if isinstance(_value, list):
                    _value = ['{}'.format('{' + ':.{}f'.format(num_digit[_attr]) + '}').format(f) for f in _value]
                    _value = ' '.join(list(map(str, _value)))
                elif isinstance(_value, dict) is True:
                    if ('vt' not in _value.keys()) and ('vn' not in _value.keys()):
                        # f consists of only v.
                        _value = ' '.join(list(map(str, _value['v'])))
                    elif ('vt' in _value.keys()) and ('vn' in _value.keys()):
                        # f consists of v, vn and vt.
                        _value = ' '.join(['/'.join(list(map(str, _x))) for _x in (np.asarray([_x for _x in _value.values()]).T).tolist()])
                    else:
                        raise NotImplementedError
                else:
                    pass

                if isinstance(_value, str) is False:
                    raise ValueError('The type of the _value should be str.')

                if _value[-1] != '\n':
                    _value += '\n'

                _value = '{} {}'.format(_attr, _value)

                if isinstance(_attributes_written, str):
                    if _attributes_written == 'all':
                        temp.append(_value)
                    else:
                        raise NotImplementedError('The _attributes_written, {} has not been supported yet.')
                elif isinstance(_attributes_written, set):
                    if _attr in _attributes_written:
                        temp.append(_value)
                else:
                    raise ValueError('The _attributes_written may be incorrect.')

        text_obj.write_lines(_list_str=temp)

    def _read_obj(self) -> list:
        return text_.TEXT(_spath_filename=self.path_obj.str, _mode='r').read_lines()

    def _parse(self, _data_str: list) -> dict:
        res = dict()
        for _idx, _line in enumerate(_data_str):
            line = _line.rstrip()
            attribute = line.split()[0]
            self._add_attribute(_attribute=attribute, _line=line, _dict=res)

        return res

    def _add_attribute(self, _attribute: str, _line: str, _dict: dict) -> None:
        item = _line.replace('{} '.format(_attribute), '')
        if _attribute in {'#', 'mtllib', 'o', 's', 'usemtl'}:
            pass
        elif _attribute in {'v', 'vn', 'vt'}:
            item = list(map(float, item.split()))
        elif _attribute in {'f'}:
            if '/' in item:
                item = {
                    'v': [int(_x.split('/')[0]) for _x in item.split()],
                    'vt': [int(_x.split('/')[1]) for _x in item.split()],
                    'vn': [int(_x.split('/')[2]) for _x in item.split()]
                }
            else:
                item = {
                    'v': [int(_x) for _x in item.split()]
                }
        else:
            raise NotImplementedError('The given attribute, {} has not been supported yet.'.format(_attribute))

        if _attribute not in _dict.keys():
            _dict[_attribute] = list()
        _dict[_attribute].append(item)


class FunctionVertex(object):
    @staticmethod
    def read(_spath_obj: str) -> np.ndarray:
        path_obj = path_.Path(_spath_obj)

        if (path_obj.path.is_file() and path_obj.path.exists()):
            temp = list()

            with open(path_obj.str, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith('v '):
                        temp.append(list(map(float, line.rstrip().split(' ')[1:])))
        else:
            raise FileNotFoundError('The given _spath_file, {} is not existed.'.format(_spath_obj))

        return np.asarray(temp)

    @staticmethod
    def translate(_ndarr_v: np.ndarray, _ndarr_offset: np.ndarray) -> np.ndarray:
        offset = _ndarr_offset

        return _ndarr_v + offset

    @staticmethod
    def get_midpoint(_ndarr_v: np.ndarray) -> np.ndarray:
        return np.mean(_ndarr_v, axis=0)

    @staticmethod
    def get_volume_cube_3d(_ndarr_v: np.ndarray) -> float:
        res = 1.0
        max_pts = list(map(float, np.max(_ndarr_v, axis=0)))
        min_pts = list(map(float, np.min(_ndarr_v, axis=0)))
        distance_pts = [abs(_max_pts - _min_pts) for (_max_pts, _min_pts) in zip(max_pts, min_pts)]

        for _distance_pt in distance_pts:
            res *= _distance_pt

        return res

    @staticmethod
    def get_number_of_vertices(_ndarr_v: np.ndarray) -> int:
        return len(_ndarr_v)


class Vertex(object):
    num_dimension = 3

    def __init__(self, _ndarr_v: np.ndarray) -> None:
        super(Vertex, self).__init__()
        self.ndarr_v = _ndarr_v
        self.num_v = None
        self.midpoint = None
        self.volume = None
        self._check_dimension(_ndarr_v=self.ndarr_v)
        self._update()
        self.info_ori = {
            'v': self.ndarr_v,
            'num_v': self.num_v,
            'midpoint': self.midpoint,
            'volume': self.volume
        }

    @property
    def data(self) -> np.ndarray:
        return self.ndarr_v

    def add(self, _ndarr_v_to_be_added: np.ndarray, _idx_v_to_be_added: Optional[list] = None):
        if _idx_v_to_be_added is None:
            # append mode
            pass
        else:
            # add mode by _idx_v_to_be_added
            pass
        raise NotImplementedError

    def remove(self, _idx_v_to_be_removed: list, _is_keepdim: bool = False, _val_keepdim: float = 0.0) -> None:
        raise NotImplementedError

    def replace(self, _ndarr_v_new: np.ndarray, _idx_v: list) -> None:
        self._check_dimension(_ndarr_v=_ndarr_v_new)
        if _ndarr_v_new.shape[0] != len(_idx_v):
            raise ValueError('The both _ndarr_v_new.shape[0] and len(_idx_v) should be same.')
        for _idx, (_ele_v_new, _ele_idx_v) in enumerate(zip(_ndarr_v_new, _idx_v)):
            if (0 <= _ele_idx_v) and (_ele_idx_v < self.num_v):
                self.ndarr_v[_ele_idx_v] = _ele_v_new
            else:
                raise NotImplementedError('The _ele_v_new, {} is out of length.'.format(_ele_v_new))
        self._update()

    def scale(self, _scale_factor: float):
        self.ndarr_v *= _scale_factor
        self._update()

    def translate_origin(self) -> None:
        self.ndarr_v = FunctionVertex.translate(_ndarr_v=self.ndarr_v, _ndarr_offset=-np.mean(self.ndarr_v, axis=0))
        self._update()

    def translate(self, _ndarr_offset: np.ndarray) -> None:
        self.ndarr_v = FunctionVertex.translate(_ndarr_v=self.ndarr_v, _ndarr_offset=_ndarr_offset)
        self._update()

    def _update(self) -> None:
        self.num_v = FunctionVertex.get_number_of_vertices(_ndarr_v=self.ndarr_v)
        self.midpoint = FunctionVertex.get_midpoint(_ndarr_v=self.ndarr_v)
        self.volume = FunctionVertex.get_volume_cube_3d(_ndarr_v=self.ndarr_v)

    def _check_dimension(self, _ndarr_v: np.ndarray) -> None:
        if _ndarr_v.ndim != 2:
            raise ValueError('The ndarr_v.ndim should be 2.')

        if _ndarr_v.shape[1] != self.num_dimension:
            raise NotImplementedError
