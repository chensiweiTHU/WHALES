#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH

python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --version v1.0-mini \
    --extra-tag nuscenes \