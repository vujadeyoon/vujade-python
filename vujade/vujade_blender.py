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


class Blender(object):
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
