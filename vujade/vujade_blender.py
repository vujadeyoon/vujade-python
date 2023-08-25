"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_blender.py
Description: A module for Blender

Usage:
    i)  blender --background --python ${path_python} -- ${arguments}
    ii) cat ${path_python}
        if __name__=='__main__':
            argv = sys.argv
            argv = argv[argv.index("--") + 1:] # argv[0]: ${arguments}
"""


import bpy
import numpy as np
from typing import Optional, Tuple, Union


class Blender(object):
    @staticmethod
    def load_image(_spath_img: str, _is_remove_alpha_channel: bool = True, _is_convert_cv2_format: bool = True) -> np.ndarray:
        if (_is_remove_alpha_channel is False) and (_is_convert_cv2_format is True):
            raise NotImplementedError('The _is_remove_alpha_channel should be True when _is_convert_cv2_format is True.')

        bpy_image = bpy.data.images.load(_spath_img, check_existing=False)
        img_width, img_height = bpy_image.size[0], bpy_image.size[1]
        data = np.asarray(bpy_image.pixels[:])
        if len(bpy_image.pixels[:]) == (4 * img_width * img_height):
            res = data.reshape(img_height, img_width, 4)
        else:
            raise ValueError('The bpy_image format should be RGBA.')

        if _is_remove_alpha_channel is True:
            res = res[:, :, :3]

        if _is_convert_cv2_format is True:
            res *= 255.0
            res = res[::-1, :, :]
            res = res[:, :, ::-1] # RGB to BGR
            res = np.clip(res, 0.0, 255.0).astype(np.uint8)

        return res

    @staticmethod
    def deselect_all() -> None:
        bpy.ops.object.select_all(action='DESELECT')

    @staticmethod
    def select_objects(_names: Tuple[str], _is_deselect_all: bool = True) -> None:
        if isinstance(_names, str):
            raise ValueError('The _names, {} should be Tuple[str], not str.'.format(_names))

        if _is_deselect_all is True:
            Blender.deselect_all()

        for _name in _names:
            obj = Blender.get_object(_name=_name)
            if len(_names) == 1:
                bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

    @staticmethod
    def export_fbx(
        _spath_fbx: str,
        _use_selection: Union[bool, Tuple[str]] = False,
        _use_visible: bool = False,
        _path_mode: str = 'AUTO',
        _embed_textures: bool = False,
        _axis_forward: str ='-Z',
        _axis_up: str = 'Y'
    ) -> None:
        if _path_mode not in {'AUTO', 'ABSOLUTE', 'RELATIVE', 'MATCH', 'STRIP', 'COPY'}:
            raise ValueError('The _path_mode, {} has not been supported.'.format(_path_mode))
        if _axis_forward not in {'X', 'Y', 'Z', '-X', '-Y', '-Z'}:
            raise ValueError('The _axis_forward, {} has not been supported.'.format(_axis_forward))
        if _axis_up not in {'X', 'Y', 'Z', '-X', '-Y', '-Z'}:
            raise ValueError('The _axis_up, {} has not been supported.'.format(_axis_up))

        if isinstance(_use_selection, tuple):
            Blender.select_objects(_names=_use_selection, _is_deselect_all=True)
            _use_selection = True

        bpy.ops.export_scene.fbx(
            filepath=_spath_fbx,
            use_selection=_use_selection,
            use_visible=_use_visible,
            path_mode=_path_mode,
            embed_textures=_embed_textures,
            axis_forward=_axis_forward,
            axis_up=_axis_up
        )

    @staticmethod
    def export_obj(
        _spath_obj: str,
        _use_selection: Union[bool, Tuple[str]] = False,
        _use_normals: bool = True,
        _use_uvs: bool = True,
        _use_materials: bool = True,
        _use_triangles: bool = False,
        _keep_vertex_order: bool = False,
        _global_scale: float = 1.0,
        _path_mode: str = 'AUTO',
        _axis_forward='-Z',
        _axis_up='Y'
    ) -> None:
        if _path_mode not in {'AUTO', 'ABSOLUTE', 'RELATIVE', 'MATCH', 'STRIP', 'COPY'}:
            raise ValueError('The _path_mode, {} has not been supported.'.format(_path_mode))
        if _axis_forward not in {'X', 'Y', 'Z', '-X', '-Y', '-Z'}:
            raise ValueError('The _axis_forward, {} has not been supported.'.format(_axis_forward))
        if _axis_up not in {'X', 'Y', 'Z', '-X', '-Y', '-Z'}:
            raise ValueError('The _axis_up, {} has not been supported.'.format(_axis_up))

        if isinstance(_use_selection, tuple):
            Blender.select_objects(_names=_use_selection, _is_deselect_all=True)
            _use_selection = True

        bpy.ops.export_scene.obj(
            filepath=_spath_obj,
            use_selection=_use_selection,
            use_normals=_use_normals,
            use_uvs=_use_uvs,
            use_materials=_use_materials,
            use_triangles=_use_triangles,
            keep_vertex_order=_keep_vertex_order,
            global_scale=_global_scale,
            path_mode=_path_mode,
            axis_forward=_axis_forward,
            axis_up=_axis_up
        )

    @staticmethod
    def get_objects_from_scene():
        return bpy.context.scene.objects

    @staticmethod
    def get_name_objects_from_scene() -> set:
        res = set()
        scene_objects = Blender.get_objects_from_scene()
        for _idx, _obj in enumerate(scene_objects):
            res.add(_obj.name)
        return res

    @staticmethod
    def get_object(_name: str):
        try:
            res = bpy.data.objects[_name]
        except Exception as e:
            raise KeyError("The object, '{}' should not be existed in the scene (i.e. {})".format(_name, Blender.get_name_objects_from_scene()))
        return res

    @staticmethod
    def remove_objects(_names: Optional[Tuple[str]] = ('Cube', 'Light', 'Camera')) -> None:
        scene_objects = Blender.get_objects_from_scene()
        for _idx, _obj in enumerate(scene_objects):
            if _names is None:
                # Remove all objects in the scene.
                Blender.remove_object(_name=_obj.name)
            else:
                if _obj.name in _names:
                    Blender.remove_object(_name=_obj.name)

    @staticmethod
    def remove_object(_name: str) -> None:
        try:
            Blender.deselect_all()
            Blender.get_object(_name=_name).select_set(True)
            bpy.ops.object.delete()
        except Exception as e:
            pass

    @staticmethod
    def get_shapekeys(_object) -> set:
        res = set()
        shape_keys = _object.data.shape_keys
        for _idx, _key in enumerate(shape_keys.key_blocks):
            res.add(_key.name)
        return res

    @staticmethod
    def reset_shapekeys():
        for k in bpy.data.shape_keys.keys():
            if bpy.data.shape_keys[k].key_blocks is not None :
                print(bpy.data.shape_keys[k].keys())
                for i in bpy.data.shape_keys[k].key_blocks:
                    i.value = 0

    @staticmethod
    def set_shapekey_1(_object, _name: str, _value: float) -> None:
        # _is_apply_shapekey should be True when applying ShapeKey immediately.
        is_hit = False
        shape_keys = _object.data.shape_keys

        for _idx, _key in enumerate(shape_keys.key_blocks):
            if _key.name == _name:
                is_hit = True
                _key.value = _value

        if is_hit is False:
            raise ValueError("The given ShapeKey, '{}' should not be in the shape keys of the given object (i.e. '{}').".format(_name, Blender.get_shapekeys(_object=_object)))

    @staticmethod
    def set_shapekey_2(name: str, value: float) -> None:
        for k in bpy.data.shape_keys.keys():
            if bpy.data.shape_keys[k].key_blocks is not None:
                for i in bpy.data.shape_keys[k].key_blocks:
                    if i.name == name:
                        i.value = value

    @staticmethod
    def getVertices(_object, _is_global_coordinates: bool = False) -> np.ndarray:
        """
        Reference:
            i)   https://stackoverflow.com/questions/70282889/to-mesh-in-blender-3
            ii)  https://blenderartists.org/t/get-mesh-data-with-modifiers-applied-in-2-8/1163217
            iii) https://blender.stackexchange.com/questions/1311/how-can-i-get-vertex-positions-from-a-mesh
        """
        """
        Algorithm 1: is equal to the Algorithm 2 (verified.)
        dg = bpy.context.evaluated_depsgraph_get()
        obj = _object.evaluated_get(dg)
        mesh = _object.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
        if _is_global_coordinates is True:
            verts = [obj.matrix_world @ _vert.co for _vert in mesh.vertices]
        else:
            verts = [_vert.co for _vert in mesh.vertices]
        """

        # Algorithm 2
        if _is_global_coordinates is True:
            verts = [_object.matrix_world @ _vert.co for _vert in _object.data.vertices]
        else:
            verts = [_vert.co for _vert in _object.data.vertices]

        return np.asarray([vert.to_tuple() for vert in verts])

    @staticmethod
    def remove_object_cube_default() -> None:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    @staticmethod
    def adjust_camera_sensor(_width: int, _height: int) -> None:
        bpy.context.scene.camera.data.sensor_width = _width
        bpy.context.scene.camera.data.sensor_height = _height

    @staticmethod
    def adjust_render_resolution(_x: int, _y: int, _percentage: int) -> None:
        bpy.context.scene.render.resolution_x = _x
        bpy.context.scene.render.resolution_y = _y
        bpy.context.scene.render.resolution_percentage = _percentage

    @staticmethod
    def adjust_camera(_location: tuple, _rotation: tuple) -> None:
        bpy.context.scene.camera.location = _location
        bpy.context.scene.camera.rotation_euler = _rotation

    @staticmethod
    def object_transform(_name_object: str, _location: tuple, _rotation: tuple) -> None:
        object = bpy.data.objects[_name_object]
        object.location = _location
        object.rotation_euler = _rotation

    @staticmethod
    def import_obj(_spath_obj: str) -> None:
        bpy.ops.import_scene.obj(filepath=_spath_obj)

    @staticmethod
    def move_object(_name_object: str, _collection_src, _collection_dst) -> None:
        obj = bpy.data.objects[_name_object]
        _collection_src.objects.unlink(obj)
        _collection_dst.objects.link(obj)

    @staticmethod
    def hide_object(_name_object: str, _is_hide: bool = True) -> None:
        obj = bpy.data.objects[_name_object]
        obj.hide_set(_is_hide)

    @staticmethod
    def render_object(_name_object: str, _is_render: bool = True) -> None:
        obj = bpy.data.objects[_name_object]
        obj.hide_render = _is_render

    @staticmethod
    def render(_spath_img: str) -> None:
        bpy.context.scene.render.filepath = _spath_img
        bpy.ops.render.render(write_still=True)

    @staticmethod
    def save_blend(_spath_abs_blend: str) -> None:
        bpy.ops.wm.save_as_mainfile(filepath=_spath_abs_blend)
