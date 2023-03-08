"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_wavefront.py
Description: A module for wavefront
"""


import os
import numpy as np
from typing import List, Optional, Set, Union
from vujade import vujade_list as list_
from vujade import vujade_path as path_
from vujade import vujade_text as text_
from vujade.vujade_debug import printd


class Wavefront(object):
    def __init__(self, _spath_obj: str) -> None:
        super(Wavefront, self).__init__()
        self.attr_global = {'#', 'mtllib', 'o'}
        self.attr_object = {'v', 'vn', 'vt', 's', 'usemtl', 'f'}
        self.attr_all = self.attr_global | self.attr_object
        self.name_object_none = 'None'
        self.num_digit = {
            'v': 6,
            'vn': 4,
            'vt': 6
        }

        self.path_obj = path_.Path(_spath_obj)
        self.data_obj = self._parse_obj(self._read_text(_spath_text=self.path_obj.str))

        if 'mtllib' in self.data_obj.keys():
            self.path_mtl = path_.Path(os.path.join(self.path_obj.parent.str, self.data_obj['mtllib']))
        else:
            self.path_mtl = None

    @staticmethod
    def get_volume_from_triangle_faces(_data_object: dict) -> float:
        triangle_volumes = list()
        faces = _data_object['f']

        for face in faces:
            vertex_1 = np.asarray(_data_object['v'][face['v'][0]])
            vertex_2 = np.asarray(_data_object['v'][face['v'][1]])
            vertex_3 = np.asarray(_data_object['v'][face['v'][2]])
            volume = np.dot(vertex_1, np.cross(vertex_2, vertex_3)) / 6.0
            triangle_volumes.append(volume)

        res = abs(sum(triangle_volumes))

        return res

    def get_name_objects(self) -> list:
        return list_.sorted_set(_list=list(self.data_obj['o'].keys()))

    def change_name_object(self, _name_object_src: Optional[str] = None, _name_object_dst: str = '') -> None:
        if _name_object_src is None:
            if len(self.get_name_objects()) == 1:
                name_object_src = ''.join(self.get_name_objects())
            else:
                raise ValueError('The _name_object_src should not be None when len(self.get_name_objects()) != 1.')
        else:
            name_object_src = _name_object_src
            self._check_object(_name_object=name_object_src)
        self.data_obj['o'][_name_object_dst] = self.data_obj['o'].pop(name_object_src)

    def get_object(self, _name_object: Optional[str] = None) -> dict:
        if _name_object is None:
            if len(self.get_name_objects()) == 1:
                name_object = ''.join(self.get_name_objects())
            else:
                raise ValueError('The _name_object should not be None when len(self.get_name_objects()) != 1.')
        else:
            name_object = _name_object
            self._check_object(_name_object=name_object)
        return self.data_obj['o'][name_object]

    def set_object(self, _data_object: dict, _name_object: Optional[str] = None) -> None:
        if _name_object is None:
            if len(self.get_name_objects()) == 1:
                name_object = ''.join(self.get_name_objects())
            else:
                raise ValueError('The _name_object should not be None when len(self.get_name_objects()) != 1.')
        else:
            name_object = _name_object
        self.data_obj['o'][name_object] = _data_object

    def setattr_v(self, _v: List[float], _name_object: Optional[str] = None) -> None:
        if _name_object is None:
            if len(self.get_name_objects()) == 1:
                name_object = ''.join(self.get_name_objects())
            else:
                raise ValueError('The _name_object should not be None when len(self.get_name_objects()) != 1.')
        else:
            name_object = _name_object
            self._check_object(_name_object=name_object)

        if len(_v) != len(self.data_obj['o'][name_object]['v']):
            raise ValueError("Both len(_v) and len(self.data_object['o'][{}]['v']) should be same.".format(name_object))

        self.data_obj['o'][name_object]['v'] = _v

    def write_obj(
            self,
            _spath_obj: str,
            _objects_written: Union[Set[str], str] = 'all',
            _attributes_written: Union[Set[str], str] = 'all',
            _num_digit: Optional[dict] = None,
            _attributes_append: Optional[List[str]] = None,
            _is_reorder: bool = False,
            _order_object: Union[list, tuple] = ('v', 'vn', 'vt', 's', 'usemtl', 'f')
    ) -> None:
        if _is_reorder is True:
            self._change_order(_order_object=_order_object)

        objects_written = self._get_objects_written(_objects_written=_objects_written)
        attr_written = self._get_attributes_written(_attributes_written=_attributes_written)
        num_digit = self._get_num_digit(_num_digit=_num_digit)

        text_obj = text_.TEXT(_spath_filename=_spath_obj, _mode='w')
        temp = list()

        for _idx, (_attr, _values) in enumerate(self.data_obj.items()):
            if _attr == '#':
                for _value in _values:
                    if _attr in attr_written:
                        temp.append('{} {}\n'.format(_attr, _value))
            elif _attr == 'mtllib':
                if _attr in attr_written:
                    temp.append('{} {}\n'.format(_attr, _values))
            elif _attr == 'o':
                for _idy, (_name_object, _attr_objects) in enumerate(_values.items()):
                    if _name_object in objects_written:
                        if (_name_object != self.name_object_none) and (self._is_write_attr_o(_attr_objects=_attr_objects, _attr_written=attr_written)):
                            temp.append('{} {}\n'.format(_attr, _name_object))

                        for _idz, (_attr_object, _ele_object) in enumerate(_attr_objects.items()):
                            if _attr_object in attr_written:
                                if _attr_object == 's' and isinstance(_ele_object, int):
                                    _ele_object = '{}'.format(_ele_object)

                                if _attr_object in {'v', 'vn', 'vt'} and isinstance(_ele_object, list):
                                    for _e in _ele_object:
                                        temp.append('{} {}\n'.format(_attr_object, ' '.join([self._get_str_format(_attr=_attr_object, _num_digit=num_digit).format(_f) for _f in _e])))
                                elif _attr_object in {'s', 'usemtl'} and isinstance(_ele_object, str):
                                    temp.append('{} {}\n'.format(_attr_object, _ele_object))
                                elif _attr_object in {'f'} and isinstance(_ele_object, list):
                                    for _e in _ele_object:
                                        if ('vn' not in _e.keys()) and ('vt' not in _e.keys()):
                                            if list(_e.keys()) != ['v']:
                                                raise KeyError('f consists of only v.')
                                            temp.append('{} {}\n'.format(_attr_object, ' '.join(list(map(str, _e['v'])))))
                                        elif ('vn' in _e.keys()) and ('vt' in _e.keys()):
                                            if list(_e.keys()) != ['v', 'vt', 'vn']:
                                                raise KeyError("f consists of v, vt and vn with the order. (i.e. order: dict('v': [], 'vt': [], 'vn': [])), but the given order is {}.".format(list(_e.keys())))
                                            temp.append('{} {}\n'.format(_attr_object, ' '.join(['/'.join(list(map(str, _x))) for _x in (np.asarray([_x for _x in _e.values()]).T).tolist()])))
                                        else:
                                            raise NotImplementedError('The format of the attribute, f has not been supported yet.')
                                else:
                                    raise NotImplementedError('The attribute, {} for object has not been supported yet.'.format(_attr_object))
            else:
                raise NotImplementedError('The attribute, {} has not been supported yet.'.format(_attr))

        if _attributes_append is not None:
            for _str_appended in _attributes_append:
                temp.append(_str_appended)

        text_obj.write_lines(_list_str=temp)

    def write_mtl(self, _list_str: List[str], _spath_mtl: str) -> None:
        with open(_spath_mtl, 'w') as f:
            for _ele in _list_str:
                f.write(_ele)

    def _read_text(self, _spath_text: str) -> list:
        return text_.TEXT(_spath_filename=_spath_text, _mode='r').read_lines()

    def _parse_obj(self, _data_str: list) -> dict:
        res = dict()
        name_object = self.name_object_none

        for _idx, _line in enumerate(_data_str):
            line = _line.rstrip()
            if line == '':
                continue
            else:
                attribute = line.split()[0]
                value = line.replace('{} '.format(attribute), '')
                if attribute in {'#', 'mtllib'}:
                    self._add_attribute(_dict=res, _attribute=attribute, _value=value)
                elif attribute == 'o':
                    name_object = value
                elif attribute in {'v', 'vn', 'vt', 's', 'usemtl', 'f'}:
                    # Initialized a dictionary.
                    if 'o' not in res.keys():
                        res['o'] = dict()
                    if name_object not in res['o'].keys():
                        res['o'][name_object] = dict()

                    self._add_object_attribute(_dict=res['o'][name_object], _attribute=attribute, _value=value)
                else:
                    raise NotImplementedError('The given attribute, {} has not been supported yet.'.format(attribute))

        return res

    def _parse_mtl(self, _data_str: list) -> dict:
        raise NotImplementedError('The _parse_mtl has not been implemented yet.')

    def _add_attribute(self, _dict: dict, _attribute: str, _value: str) -> None:
        if _attribute in {'mtllib', 's', 'usemtl'}:
            _dict[_attribute] = _value
        else:
            if _attribute not in _dict.keys():
                _dict[_attribute] = list()

            if _value is not None:
                _dict[_attribute].append(_value)

    def _add_object_attribute(self, _dict: dict, _attribute: str, _value: str) -> None:
        if _attribute in {'v', 'vn', 'vt'}:
            value = list(map(float, _value.split()))
        elif _attribute in {'s', 'usemtl'}:
            value = _value
        elif _attribute in {'f'}:
            if '/' in _value:
                value = {
                    'v': [int(_x.split('/')[0]) for _x in _value.split()],
                    'vt': [int(_x.split('/')[1]) for _x in _value.split()],
                    'vn': [int(_x.split('/')[2]) for _x in _value.split()]
                }
            else:
                value = {
                    'v': [int(_x) for _x in _value.split()]
                }
        else:
            raise NotImplementedError('The given attribute, {} has not been supported yet.'.format(_attribute))

        self._add_attribute(_dict=_dict, _attribute=_attribute, _value=value)

    def _check_object(self, _name_object: str) -> None:
        if _name_object not in self.data_obj['o'].keys():
            raise KeyError("The _nmae_object, {} should be in self.data_obj['o'].keys().".format(_name_object))

    def _get_str_format(self, _attr: str, _num_digit: dict) -> str:
        return '{}'.format('{' + ':.{}f'.format(_num_digit[_attr]) + '}')

    def _get_objects_written(self,  _objects_written: Union[Set[str], str] = 'all') -> Set[str]:
        if (isinstance(_objects_written, set) and not _objects_written.issubset(set(self.data_obj['o'].keys()))):
            raise ValueError("The _objects_written, {} should be subset of the self.data_obj['o'].keys() (i.e. {}).".format(_objects_written, self.get_name_objects()))
        elif isinstance(_objects_written, str) and _objects_written == 'all':
            res = set(self.data_obj['o'].keys())
        elif isinstance(_objects_written, set):
            res = _objects_written
        else:
            raise NotImplementedError('The given _objects_written, {} has not been supported yet.'.format(_objects_written))
        return res

    def _get_attributes_written(self, _attributes_written: Union[Set[str], str] = 'all') -> Set[str]:
        if isinstance(_attributes_written, str):
            if _attributes_written == 'all':
                res = self.attr_all
            else:
                raise NotImplementedError('The _attributes_written, {} has not been supported yet.')
        elif isinstance(_attributes_written, set):
            res = _attributes_written
        else:
            raise ValueError('The _attributes_written may be incorrect.')
        return res

    def _change_order(self, _order_object: Union[list, tuple] = ('v', 'vn', 'vt', 's', 'usemtl', 'f')) -> None:
        res = dict()
        temp = {'o': dict()}

        # Copy attributes, # and mtllib.
        self._copy_attributes(_attr='#', _data_src=self.data_obj, _data_dst=res)
        self._copy_attributes(_attr='mtllib', _data_src=self.data_obj, _data_dst=res)

        # Copy attribute, o.
        for _idx, (_name_object, _attr_objects) in enumerate(self.data_obj['o'].items()):
            temp['o'][_name_object] = dict()
            for _idy, _ele_order_object in enumerate(_order_object):
                self._copy_attributes(_attr=_ele_order_object, _data_src=_attr_objects, _data_dst=temp['o'][_name_object])
        self._copy_attributes(_attr='o', _data_src=temp, _data_dst=res)

        # Update self.data_obj
        self.data_obj = res

    def _copy_attributes(self, _attr: str, _data_src: dict, _data_dst: dict) -> None:
        if _attr in _data_dst.keys():
            raise KeyError('The _attr, {} should not be assigned in _data_dst'.format(_attr))
        if _attr in _data_src.keys():
            _data_dst[_attr] = _data_src[_attr]

    def _get_num_digit(self, _num_digit: Optional[dict] = None):
        if _num_digit is None:
            res = self.num_digit
        else:
            res = _num_digit
        return res

    def _is_write_attr_o(self, _attr_objects: dict, _attr_written: Set[str]) -> bool:
        if _attr_written & set(_attr_objects.keys()):
            res = True
        else:
            res = False
        return res

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
    def get_three_edges_cuboid(_ndarr_v: np.ndarray) -> np.ndarray:
        max_pts = list(map(float, np.max(_ndarr_v, axis=0)))
        min_pts = list(map(float, np.min(_ndarr_v, axis=0)))
        return np.asarray([abs(_max_pts - _min_pts) for (_max_pts, _min_pts) in zip(max_pts, min_pts)])

    @staticmethod
    def get_volume_cuboid(_ndarr_v: np.ndarray) -> float:
        ndarr_distance_pts = FunctionVertex.get_three_edges_cuboid(_ndarr_v=_ndarr_v)
        return float(np.prod(ndarr_distance_pts))

    @staticmethod
    def get_area_projected(_ndarr_v: np.ndarray, _axis_projected: int = 0) -> float:
        ndarr_distance_pts = FunctionVertex.get_three_edges_cuboid(_ndarr_v=_ndarr_v)
        return float(np.prod(np.delete(ndarr_distance_pts, _axis_projected)))

    @staticmethod
    def get_max_vertex_from_axis(_ndarr_v: np.ndarray, _axis: int = 0) -> np.ndarray:
        return _ndarr_v[np.argmax(_ndarr_v[:, _axis])]

    @staticmethod
    def get_min_vertex_from_axis(_ndarr_v: np.ndarray, _axis: int = 0) -> np.ndarray:
        return _ndarr_v[np.argmin(_ndarr_v[:, _axis])]

    @staticmethod
    def get_number_of_vertices(_ndarr_v: np.ndarray) -> int:
        return len(_ndarr_v)

    @staticmethod
    def swap_axis_yz(_ndarr_v: np.ndarray) -> np.ndarray:
        res = _ndarr_v.copy()
        if res.ndim == 1:
            res[1:3] = res[[2, 1]]
        elif res.ndim == 2:
            res[:, 1:3] = res[:, [2, 1]]
        else:
            raise NotImplementedError('The swap_axis_yz has been supported only for ndim=1 or ndim=2.')
        return res


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

    def scale(self, _scale_factor: Union[float, np.ndarray]):
        if isinstance(_scale_factor, np.ndarray):
            if _scale_factor.shape != self.ndarr_v.shape:
                raise ('The shape of the _scale_factor should be same to the shape of self.ndarr_v.')

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
        self.volume = FunctionVertex.get_volume_cuboid(_ndarr_v=self.ndarr_v)

    def _check_dimension(self, _ndarr_v: np.ndarray) -> None:
        if _ndarr_v.ndim != 2:
            raise ValueError('The ndarr_v.ndim should be 2.')

        if _ndarr_v.shape[1] != self.num_dimension:
            raise NotImplementedError
