#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: bash_clean.sh
# Description: A bash script file to erase directories and files.
#
#
readonly path_curr=$(pwd)
readonly path_parents=$(dirname "${path_curr}")
#
#
unset PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
export PYTHONPATH=$PYTHONPATH:${path_curr}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
#
#
path_to_be_erased="${1}"
#
#
python3 vujade_erase.py --path_src ${path_to_be_erased} && wipe -rfi ${path_to_be_erased}
