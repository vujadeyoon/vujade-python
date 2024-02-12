#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: _bash_function.sh
# Description: A bash script file for useful functions.
#
#
function get_date_time() {
  echo "$(date +'%y%m%d_%H%M%S')"
}
#
#
function get_path_parents() {
  local path=${1}
  echo "$(dirname "${path}")"
}
#
#
function get_file_extension() {
  local path=${1}
  echo ".${path##*.}"
}
#
#
function is_log() {
  local res extension_log
  extension_log=$(get_file_extension "${PATH_LOG}")

  if [[ ${extension_log} = ".log" ]] || [[ ${extension_log} = ".txt" ]]; then
    res=true
  else
    res=false
  fi

  echo ${res}
}
#
#
function check_LEVEL_VERBOSE() {
  local level_verboses

  level_verboses=("0" "1" "2" "3")

  if [[ ! -v LEVEL_VERBOSE ]]; then
      echom "[ERROR] The LEVEL_VERBOSE should be declared in advance."
      exit 1
  fi

  if [[ "${level_verboses[*]}" != *"${LEVEL_VERBOSE}"* ]]; then
    echom "[ERROR] The LEVEL_VERBOSE, ${LEVEL_VERBOSE} has not been supported yet"
    exit 1
  fi
}
#
#
function pause() {
  local name_script_tracedback line_number_tracedback name_func_tracedback

  name_script_tracedback=$(basename "${BASH_SOURCE[1]}")
  line_number_tracedback=$(caller | awk '{print $1}')
  name_func_tracedback="${FUNCNAME[1]}"
  header="[${name_script_tracedback}:${line_number_tracedback}]"

  if [[ ! -v LEVEL_VERBOSE || ${LEVEL_VERBOSE} -ne 0 ]]; then
    read -p "${header} ${1:-Press any key to continue...}"
  elif [[ ${LEVEL_VERBOSE} -eq 0 ]]; then
    :
  else
    echo "${header} The LEVEL_VERBOSE, ${LEVEL_VERBOSE} has not been supported yet."
    exit 1
  fi
}
#
#
function echom() {
  # echom: echo improved
  local name_script_tracedback line_number_tracedback name_func_tracedback

  name_script_tracedback=$(basename "${BASH_SOURCE[1]}")
  line_number_tracedback=$(caller | awk '{print $1}')
  name_func_tracedback="${FUNCNAME[1]}"
  header="[${name_script_tracedback}:${line_number_tracedback}]"
  asctime="[$(date +'%Y-%m-%d %H:%M:%S,%3N')]"

  if [[ ! -v LEVEL_VERBOSE || ${LEVEL_VERBOSE} -ne 0 ]]; then
    if [[ $(is_log) = "true" ]]; then
      echo "${asctime} [INFO ($$)]: ${header} ${1}" | tee -a ${PATH_LOG}
    else
      echo "${header} ${1}"
    fi
  elif [[ ${LEVEL_VERBOSE} -eq 0 ]]; then
    :
  else
    echo "${header} The LEVEL_VERBOSE, ${LEVEL_VERBOSE} has not been supported yet."
    exit 1
  fi
}
#
#
function python3m() {
  local path_python3="${1}"
  local command_python3="CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1"

  if [[ ! -v LEVEL_VERBOSE || ${LEVEL_VERBOSE} -eq 3 ]]; then
    command_python3+=" python3 ${path_python3} 2>&1"
  elif [[ ${LEVEL_VERBOSE} -eq 0 || ${LEVEL_VERBOSE} -eq 1 ]]; then
    command_python3+=" python3 ${path_python3} > /dev/null 2>&1"
  elif [[ ${LEVEL_VERBOSE} -eq 2 ]]; then
    command_python3+=" python3 -W ignore ${path_python3} 2>&1"
  else
    :
  fi

  if [[ $(is_log) = "true" ]]; then
    command_python3+=" | tee -a ${PATH_LOG}"
  fi

  eval "${command_python3}"
}
#
#
if [[ $(is_log) = "true" ]]; then
  rm -rf ${PATH_LOG}
  mkdir -p "$(get_path_parents "${PATH_LOG}")"
fi
#
#
check_LEVEL_VERBOSE
#
#
echom "The vujade_func.sh is called with the LEVEL_VERBOSE, ${LEVEL_VERBOSE}."
