#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: bash_clean.sh
# Description: A bash script file to remove temporary directories.
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
#
#
find ${path_curr}/ -name __pycache__ -exec rm -rf {} \;
find ${path_curr}/ -name .idea -exec rm -rf {} \;
