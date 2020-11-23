#!/bin/bash
#
#
path_base=$(pwd)
path_cython_utils=${path_base}/vujade/utils
path_cython_nms=${path_cython_utils}/NMS/cython_nms
path_cython_scd=${path_cython_utils}/SceneChangeDetection
#
#
cd ${path_cython_nms} && rm -rf build ./*nms*.c ./*nms*.so
cd ${path_cython_scd} && rm -rf build ./*scd*.c ./*scd*.so
#
#
cd ${path_cython_nms} && python3 setup.py build_ext --inplace
cd ${path_cython_scd} && python3 setup.py build_ext --inplace
