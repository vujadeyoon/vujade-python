#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: bash_run.sh
# Description: A main bash script file to run a python3 script.
#
#
# Usage: bash ./script/bash_run.sh ${LEVEL_VERBOSE} ${PATH_LOG} ${LEVEL_LOG} ...
#
#
readonly path_curr=$(pwd)
readonly path_parents=$(dirname "${path_curr}")
readonly path_bash_function="${path_curr}/script/_bash_function.sh"
readonly name_shell_main=$(basename "${0}")
#
#
unset PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED LEVEL_VERBOSE PATH_LOG LEVEL_LOG
export PYTHONPATH=$PYTHONPATH:${path_curr}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export LEVEL_VERBOSE="${1:-1}"                                    # 0: silent; 1: summary (i.e. only shell); 2: details w/o warnings (i.e. all w/o warnings); 3: details (i.e. all).
export PATH_LOG="${2:-"./logs/log_$(date +'%y%m%d_%H%M%S').log"}" # If the PATH_LOG.suffix is '.log' or '.txt', the log will be saved (i.e. logging mode).
export LEVEL_LOG="${3:-"DEBUG"}"                                  # INFO; DEBUG; WARNING; ERROR; CRITICAL
#
#
if [[ -f "${path_bash_function}" ]]; then
  # shellcheck source=script/_bash_function.sh
  source "${path_bash_function}"
else
  echo "[${name_shell_main}:${LINENO}] The ${path_bash_function} should be prepared."
  exit 1
fi
#
#
# Custom codes are below.
#
#
python3m "${path_curr}/main.py"
