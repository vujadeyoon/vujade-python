#!/bin/bash
#
#
# Dveloper: vujadeyoon
# E-mail: sjyoon1671@gmail.com
# Github: https://github.com/vujadeyoon/vujade
#
# Title: bash_jpeg2jpg.sh
# Description: A bash script file to convert a file extension from .jpeg to .jpg.
#
#
for f in *.jpeg; do
    mv -- "$f" "${f%.jpeg}.jpg"
done
