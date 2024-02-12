#!/bin/bash
#
#
# Developer: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: bash_s3_del.sh
# Description: A bash script file to delete a file in the AWS S3.
# Recommendation: Referring to the https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/install-cliv2-linux.html#cliv2-linux-install,
#                 I recommend that you should install the AWS Command Line Interface version 2 (i.e. awscli2).
#
# Usage: bash ./vujade/bash/bash_s3_del.sh <NAME_FILE> <NAME_BUCKET> <PATH_REMOTE_BASE>
#
#
path_curr=$(pwd)
path_parent=$(dirname ${path_curr})
path_home=${HOME}
#
#
# You should define below default variables if you want to use this code conveniently.
DEFAULT_NAME_BUCKET=NAME_BUCKET
DEFAULT_PATH_REMOTE_BASE=PATH_REMOTE_BASE
#
#
path_aws=${path_home}/.aws/
path_config=${path_aws}/config
path_credentials=${path_aws}/credentials
#
#
aws_region=$(awk '/region/{print $3}' ${path_config})
aws_output=$(awk '/output/{print $3}' ${path_config})
aws_access_key_id=$(awk '/aws_access_key_id/{print $3}' ${path_credentials})
aws_secret_access_key=$(awk '/aws_secret_access_key/{print $3}' ${path_credentials})
#
#
filename=$1
#
#
mode=delete
name_bucket=${2:-${DEFAULT_NAME_BUCKET}}
path_remote_base=${3:-${DEFAULT_PATH_REMOTE_BASE}}
path_remote=${path_remote_base}/${filename}
path_local=${path_curr}/${filename}
#
#
python3 ${path_curr}/vujade/vujade_aws.py --mode ${mode} \
                                          --path_remote ${path_remote} \
                                          --path_local ${path_local} \
                                          --name_bucket ${name_bucket} \
                                          --access_key ${aws_access_key_id} \
                                          --secret_key ${aws_secret_access_key} \
                                          --path_aws ${path_aws} \
                                          --region ${aws_region} \
                                          --output ${aws_output}
#
#
echo "[bash_s3_del.sh] Finish to delete the S3://${name_bucket}/${path_remote}."
