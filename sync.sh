#!/usr/bin/env bash

rsync -avz  --exclude=__pycache__ --exclude=".*" --exclude="data/model" . gpu-sd:~/akshit.jain/repos/HRED-tensorflow