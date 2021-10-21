#!/bin/bash
#
#
# Dveloper: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: bash_setup_vujade.sh
# Description: A bash script file for installing vujade
#
#
path_base=$(pwd)
path_root=$(dirname ${path_base})
path_cython_utils=${path_base}/vujade/utils
path_cython_nms=${path_cython_utils}/NMS/cython_nms
path_cython_scd_inter=${path_cython_utils}/SceneChangeDetection/InteractiveProcessing
path_cython_scd_batch=${path_cython_utils}/SceneChangeDetection/BatchProcessing
path_cython_distance=${path_cython_utils}/Distance
#
#
find ./ -name __pycache__ -exec rm -rf {} \;
find ./ -name .idea -exec rm -rf {} \;
cd ${path_cython_distance} && rm -rf build ./*distance*.c ./*distance*.so
cd ${path_cython_nms} && rm -rf build ./*nms*.c ./*nms*.so
cd ${path_cython_scd_inter} && rm -rf build ./*scd*.c ./*scd*.so
cd ${path_cython_scd_batch} && rm -rf build ./*scd*.c ./*scd*.so
#
#
cd ${path_cython_distance} && python3 setup.py build_ext --inplace
cd ${path_cython_nms} && python3 setup.py build_ext --inplace
cd ${path_cython_scd_inter} && python3 setup.py build_ext --inplace
cd ${path_cython_scd_batch} && python3 setup.py build_ext --inplace
#
#
mv ${path_root}/vujade ${path_root}/vujade_temp/
mv ${path_root}/vujade_temp/vujade ${path_root}/
rm -rf ${path_root}/vujade_temp/
