"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_pyvista.py
Description: A module for 3D visualization

Acknowledgement:
    1. https://towardsdatascience.com/python-libraries-for-mesh-and-point-cloud-visualization-part-1-daa2af36de30
"""


import pyvista as pv
import pyvistaqt as pvqt
from typing import Optional
from vujade import vujade_path as path_
from vujade.vujade_debug import printd


class _PyVista(object):
    def __init__(self, _window_dsize: tuple = (1920, 1080)) -> None:
        super(_PyVista, self).__init__()
        self.mesh = list()
        self.texture = list()
        self.p = pvqt.BackgroundPlotter(window_size=_window_dsize)

    def add_mesh(self, _spath_obj: str, _spath_texture: Optional[str] = None, _name: str = 'name_mesh') -> None:
        path_obj = path_.Path(_spath_obj)
        self.mesh.append(self._read(_spath_file=path_obj.str))
        self.texture.append(None) if _spath_texture is None else self.texture.append(self._read_texture(_spath_file=_spath_texture))
        self.p.add_mesh(self.mesh[-1], name=_name, texture=self.texture[-1])

    def set_camera(self, _position: str = 'xy', _clipping_range: tuple = (0, 10)) -> None:
        self.p.camera_position = _position
        self.p.camera.clipping_range = _clipping_range

    def get_sphere(self, _radius: float, _center: tuple) -> pv.core.pointset.PolyData:
        return pv.Sphere(radius=_radius, center=_center)

    def add_light(self, _position: tuple, _focal_point: tuple, _color: str = 'blue') -> None:
        self.p.add_light(pv.Light(position=(0, 0, 1), focal_point=(0, 0, 0), color=_color))

    def show(self):
        self.p.show()
        self.p.app.exec_()

    def add_callback(self, **kwargs) -> None:
        """
        def add_callback(self, _interval: int) -> None:
            self.p.add_callback(self._update_scene, interval=_interval)
        """
        raise NotImplementedError('The add_callback should be implemented in drived class.')

    def add_slider_widget(self, **kwargs):
        """
        def add_slider_widget(self, _idx: int, _rng: tuple, _title: str):
            self.p.add_slider_widget(callback=self._update_scene, rng=_rng, title=_title)
        """
        raise NotImplementedError('The add_slider_widget should be implemented in drived class.')

    def _read(self, _spath_file: Optional[str] = None) -> Optional[pv.core.pointset.DataSet]:
        if _spath_file is None:
            res = None
        else:
            res = pv.get_reader(_spath_file).read()
        return res

    def _read_texture(self, _spath_file: Optional[str] = None) -> Optional[pv.core.objects.Texture]:
        if _spath_file is None:
            res = None
        else:
            res = pv.read_texture(_spath_file)
        return res

    def _update_scene(self, **kwargs) -> None:
        raise NotImplementedError('This method should be implemented in the derived class.')
