#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH

python tools/create_data.py dolphins \
    --root-path ./data/dolphins \
    --out-dir ./data/dolphins \
    --extra-tag dolphins \