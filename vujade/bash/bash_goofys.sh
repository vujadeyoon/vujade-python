#!/bin/bash
#
#
# Dveloper: vujadeyoon
# Email: vujadeyoon@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: bash_goofys.sh
# Description: A bash script file to install the goofys.
# Usage:
#     i)   Fill in the both <SECRET> in the file, ~./aws/credentials.
#          [default]
#          aws_access_key_id = <SECRET>
#          aws_secret_access_key = <SECRET>
#     ii)  Install: bash bash_goofys.sh
#     iii) Search all buckets in the AWS S3: aws s3 ls
#     iv)  Mount AWS S3:
#          - goofys ${name_bucket} ${path_mountpoint}
#          - goofys ${name_bucket}:${name_prefix} ${path_mountpoint} # if you only want to mount objects under a ${name_prefix}.
#     v)   Unmount AWS S3: umount ${path_mountpoint}
#
#
wget https://github.com/kahing/goofys/releases/download/v0.24.0/goofys
sudo cp ./goofys /usr/local/bin/goofys
sudo chmod +x /usr/local/bin/goofys
