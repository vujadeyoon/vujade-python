#!/bin/bash
#
#
# Dveloper: vujadeyoon
# E-mail: sjyoon1671@gmail.com
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
#     iii) Mount AWS S3:
#          - goofys ${name_bucket} ${path_mountpoint}
#          - goofys ${name_bucket}:${name_prefix} ${path_mountpoint} # if you only want to mount objects under a ${name_prefix}.
#     iv)  Unmount AWS S3: umount ${path_mountpoint}
#
#
wget https://github.com/kahing/goofys/releases/download/v0.24.0/goofys
sudo cp ./goofys /usr/local/bin/goofys
sudo chmod +x /usr/local/bin/goofys
