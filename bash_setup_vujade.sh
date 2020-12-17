#!/bin/bash
#
#
# Dveloper: vujadeyoon
# E-mail: sjyoon1671@gmail.com
# Github: https://github.com/vujadeyoon/vujade
# Date: Dec. 17, 2020.
#
# Title: bash_setup_vujade.sh
# Version: 0.2.0
# Description: A bash script file for installing vujade
#
#
path_base=$(pwd)
path_cython_utils=${path_base}/vujade/utils
path_cython_nms=${path_cython_utils}/NMS/cython_nms
path_cython_scd=${path_cython_utils}/SceneChangeDetection
path_cython_distance=${path_cython_utils}/Distance
#
#
cd ${path_cython_nms} && rm -rf build ./*nms*.c ./*nms*.so
cd ${path_cython_scd} && rm -rf build ./*scd*.c ./*scd*.so
cd ${path_cython_distance} && rm -rf build ./*distance*.c ./*distance*.so
#
#
cd ${path_cython_nms} && python3 setup.py build_ext --inplace
cd ${path_cython_scd} && python3 setup.py build_ext --inplace
cd ${path_cython_distance} && python3 setup.py build_ext --inplace
